#!/usr/bin/env python3
"""
BDD100K MoE router robustness evaluation under PGD transfer-based attack.

This mirrors the BDD100K decision/score attack evaluation outputs. Because
BDD100K has no ground-truth expert label, the surrogate is trained with pseudo
labels from the target router's clean top-1 expert. PGD is then run on the
surrogate only, and the resulting images are evaluated on the real MoE.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DEFAULT_DECISION_SCRIPT = Path(__file__).resolve().with_name("bdd100k_moe_decision_attack.py")
DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_TEST_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_test.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/data/metadata_files/metadata-New"
DEFAULT_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/New-train-models/moe_stage3"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Evaluation/BDD100k/transfer_attack_outputs/PGD"

EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}
NUM_ROUTER_CLASSES = 3


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def install_optional_foolbox_stub() -> None:
    """Import shared decision helpers without requiring Foolbox/numba."""
    fb = types.ModuleType("foolbox")
    fb.__file__ = "<transfer_attack_foolbox_stub>"
    fb.__package__ = "foolbox"
    fb.__spec__ = importlib.util.spec_from_loader("foolbox", loader=None)

    class _UnavailablePyTorchModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Foolbox is unavailable; decision-based attacks cannot run.")

    class _UnavailableAttack:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Foolbox is unavailable; decision-based attacks cannot run.")

    fb.PyTorchModel = _UnavailablePyTorchModel
    fb.attacks = types.SimpleNamespace(HopSkipJumpAttack=_UnavailableAttack, BoundaryAttack=_UnavailableAttack)
    sys.modules["foolbox"] = fb


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_decision_helpers(path: Path):
    install_optional_foolbox_stub()
    return import_module_from_path("bdd100k_moe_decision_attack_transfer_shared", path.resolve())


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SurrogateRouterSmall(nn.Module):
    """Architecturally different surrogate router for 3-way expert selection."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, NUM_ROUTER_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class PseudoRouterLabelDataset(Dataset):
    def __init__(self, base_dataset: Dataset, labels: torch.Tensor) -> None:
        self.base_dataset = base_dataset
        self.labels = labels.long().cpu()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.base_dataset[index]
        return item["image"], self.labels[index]


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    return clamp_01(torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps))


@torch.no_grad()
def collect_router_labels(
    router: nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_label: str,
) -> torch.Tensor:
    labels: List[torch.Tensor] = []
    router.eval()
    total_batches = len(loader)
    seen = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        seen += int(images.shape[0])
        print(
            f"[{timestamp()}] {progress_label}: batch {batch_idx}/{total_batches} "
            f"({seen}/{len(loader.dataset)} images)",
            flush=True,
        )
        labels.append(torch.argmax(router(images), dim=1).detach().cpu())
    return torch.cat(labels, dim=0).long()


def train_surrogate(
    surrogate: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    log_path: Optional[Path],
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    surrogate.train()
    optimizer = optim.AdamW(surrogate.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        start = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = surrogate(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            n = int(images.shape[0])
            total_loss += float(loss.item()) * n
            correct += int((torch.argmax(logits.detach(), dim=1) == labels).sum().item())
            total += n

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total),
            "accuracy": correct / max(1, total),
            "seconds": time.perf_counter() - start,
        }
        rows.append(row)
        if log_path is not None:
            write_csv(log_path, rows)
        print(
            f"[{timestamp()}] surrogate epoch {epoch}/{epochs}: "
            f"loss={row['loss']:.5f}, acc={row['accuracy']:.4f}",
            flush=True,
        )
    surrogate.eval()


def pgd_linf_untargeted(
    surrogate: nn.Module,
    x: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    random_start: bool,
    restarts: int,
) -> torch.Tensor:
    surrogate.eval()
    x_orig = x.detach()
    n = int(x_orig.shape[0])
    best_x = x_orig.clone()
    best_loss = torch.full((n,), -1e9, device=x_orig.device)

    for restart in range(max(1, restarts)):
        if random_start or restart > 0:
            x_adv = project_linf(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps), x_orig, eps)
        else:
            x_adv = x_orig.clone()

        for _step in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = surrogate(x_adv)
            loss_per_sample = nn.functional.cross_entropy(logits, y_ref, reduction="none")
            loss = loss_per_sample.sum()
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = project_linf(x_adv + step_size * torch.sign(grad), x_orig, eps)

        with torch.no_grad():
            loss = nn.functional.cross_entropy(surrogate(x_adv), y_ref, reduction="none")
            select = loss > best_loss
            if bool(select.any()):
                best_x[select] = x_adv.detach()[select]
                best_loss[select] = loss[select]
    return best_x.detach()


def craft_transfer_adversarial_images(
    surrogate: nn.Module,
    target_router: nn.Module,
    loader: DataLoader,
    clean_router_labels: torch.Tensor,
    device: torch.device,
    eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    pgd_random_start: bool,
    pgd_restarts: int,
    progress_label: str,
) -> Tuple[torch.Tensor, np.ndarray]:
    adv_chunks: List[torch.Tensor] = []
    success_chunks: List[np.ndarray] = []
    offset = 0
    total_batches = len(loader)
    target_router.eval()

    for batch_idx, batch in enumerate(loader, start=1):
        xb = batch["image"].to(device, non_blocking=True)
        n = int(xb.shape[0])
        yb = clean_router_labels[offset : offset + n].to(device, non_blocking=True)
        print(
            f"[{timestamp()}] {progress_label}: batch {batch_idx}/{total_batches} "
            f"({offset + n}/{len(loader.dataset)} images)",
            flush=True,
        )
        offset += n

        x_adv = pgd_linf_untargeted(
            surrogate=surrogate,
            x=xb,
            y_ref=yb,
            eps=eps,
            steps=pgd_steps,
            step_size=pgd_step_size,
            random_start=pgd_random_start,
            restarts=pgd_restarts,
        )
        with torch.no_grad():
            adv_top1 = torch.argmax(target_router(x_adv), dim=1)
        success = adv_top1.ne(yb)

        adv_chunks.append(x_adv.detach().cpu())
        success_chunks.append(success.detach().cpu().numpy().astype(np.bool_).reshape(-1))

    return torch.cat(adv_chunks, dim=0), np.concatenate(success_chunks, axis=0)


def aggregate_and_write(eval_module, csv_dir: Path, prefix: str, rows: List[Dict[str, Any]], key_cols: Sequence[str]) -> None:
    eval_module.write_csv(csv_dir / f"{prefix}_aggregate.csv", eval_module.summarize_table(rows, key_cols=key_cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="BDD100K MoE PGD transfer-based router attack evaluation")
    parser.add_argument("--decision-script", type=str, default=str(DEFAULT_DECISION_SCRIPT))
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--test-script", type=str, default=DEFAULT_TEST_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--run-glob", type=str, default=DEFAULT_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sur-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sur-epochs", type=int, default=50)
    parser.add_argument("--sur-lr", type=float, default=1e-3)
    parser.add_argument("--sur-weight-decay", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=0.09, help="L-infinity budget in pixel space [0, 1].")
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument("--pgd-step-size", type=float, default=0.0, help="Use 0 to auto-scale as eps / pgd_steps * 2.5.")
    parser.add_argument("--pgd-restarts", type=int, default=3)
    parser.add_argument("--no-pgd-random-start", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    if args.pgd_step_size <= 0:
        args.pgd_step_size = (float(args.eps) / max(1, int(args.pgd_steps))) * 2.5
        print(f"[{timestamp()}] PGD step size auto-scaled to {args.pgd_step_size:.6f}", flush=True)

    shared = import_decision_helpers(Path(args.decision_script))
    shared.ensure_train_import_dependencies()
    train_module = shared.import_module_from_path("bdd100k_moe_train_transfer_attack_module", Path(args.train_script).resolve())
    eval_module = shared.import_module_from_path("bdd100k_moe_test_transfer_attack_module", Path(args.test_script).resolve())

    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    ensure_dir(csv_dir)
    ensure_dir(plots_dir)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    attack_name = "transfer_pgd"
    pgd_random_start = not args.no_pgd_random_start

    bundle = shared.load_json(metadata_dir / "metadata_bundle.json")
    train_rows = shared.load_json(metadata_dir / "train.json")
    eval_rows = shared.load_json(metadata_dir / "test_fixed.json")
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_samples > 0:
        eval_rows = eval_rows[: args.max_samples]

    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    run_dirs = shared.list_run_dirs(args.run_dirs, args.run_glob)

    print(
        f"[{timestamp()}] Evaluating {len(run_dirs)} run(s), "
        f"{len(train_rows)} surrogate-train image(s), {len(eval_rows)} eval image(s).",
        flush=True,
    )
    print(
        f"[{timestamp()}] Attack={attack_name}, eps={args.eps}, "
        f"pgd_steps={args.pgd_steps}, random_start={pgd_random_start}",
        flush=True,
    )

    all_tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    robustness_rows: List[Dict[str, Any]] = []
    degradation_rows: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        print(f"[{timestamp()}] Run: {run_dir}", flush=True)
        model, config = shared.load_model_for_run(run_dir, args.checkpoint_name, train_module, bundle, device)
        normalized_moe = shared.NormalizedMoE(model).to(device).eval()
        router = shared.RouterOnly(normalized_moe).to(device).eval()

        image_size = int(config.get("image_size", 224))
        train_pixel_dataset = shared.BDD100KPixelDataset(train_rows, image_size=image_size)
        eval_pixel_dataset = shared.BDD100KPixelDataset(eval_rows, image_size=image_size)
        train_label_loader = shared.make_loader(train_pixel_dataset, args.batch_size, args.num_workers, device)
        clean_loader = shared.make_loader(eval_pixel_dataset, args.batch_size, args.num_workers, device)

        train_router_labels = collect_router_labels(
            router,
            train_label_loader,
            device,
            progress_label=f"{run_dir.name} surrogate pseudo labels",
        )
        surrogate_dataset = PseudoRouterLabelDataset(train_pixel_dataset, train_router_labels)
        surrogate = SurrogateRouterSmall().to(device)
        seed = int(config.get("seed", -1))
        train_surrogate(
            surrogate=surrogate,
            dataset=surrogate_dataset,
            device=device,
            epochs=int(args.sur_epochs),
            batch_size=int(args.sur_batch_size),
            num_workers=int(args.num_workers),
            lr=float(args.sur_lr),
            weight_decay=float(args.sur_weight_decay),
            log_path=None,
        )

        clean_router_labels = collect_router_labels(
            router,
            clean_loader,
            device,
            progress_label=f"{run_dir.name} clean router labels",
        )
        adv_images, success = craft_transfer_adversarial_images(
            surrogate=surrogate,
            target_router=router,
            loader=clean_loader,
            clean_router_labels=clean_router_labels,
            device=device,
            eps=float(args.eps),
            pgd_steps=int(args.pgd_steps),
            pgd_step_size=float(args.pgd_step_size),
            pgd_random_start=pgd_random_start,
            pgd_restarts=int(args.pgd_restarts),
            progress_label=f"{run_dir.name} transfer PGD attack",
        )

        adv_dataset = shared.DictTensorDataset(adv_images, eval_pixel_dataset)
        adv_loader = shared.make_loader(adv_dataset, args.batch_size, args.num_workers, device)

        clean_preds = shared.collect_predictions(
            normalized_moe,
            clean_loader,
            device,
            amp_dtype,
            progress_label=f"{run_dir.name} clean evaluation",
        )
        adv_preds = shared.collect_predictions(
            normalized_moe,
            adv_loader,
            device,
            amp_dtype,
            progress_label=f"{run_dir.name} adversarial evaluation",
        )

        clean_tables = shared.compute_eval_tables(
            clean_preds, eval_module, weather_names, scene_names, time_names,
            args.threshold, seed, run_dir, attack_name, "clean",
        )
        adv_tables = shared.compute_eval_tables(
            adv_preds, eval_module, weather_names, scene_names, time_names,
            args.threshold, seed, run_dir, attack_name, "adv",
        )

        for key, value in clean_tables.items():
            all_tables[f"clean_{key}"].extend(value)
        for key, value in adv_tables.items():
            all_tables[f"adv_{key}"].extend(value)

        robustness_rows.extend(shared.routing_flip_rows(clean_preds, adv_preds, success, seed, run_dir, attack_name))
        degradation_rows.extend(shared.degradation_rows(clean_tables, adv_tables, seed, run_dir, attack_name))
        margin_rows.extend(shared.margin_plot_rows_and_files(clean_preds, adv_preds, plots_dir, seed, run_dir, attack_name))
        shared.fused_attack_comparison_plots(adv_preds, eval_module, plots_dir, seed, run_dir, attack_name)

        transfer_rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack_name,
                "n": int(success.size),
                "target_router_flip_rate": float(np.mean(success)) if success.size else float("nan"),
                "target_gradient_queries": 0,
                "target_score_queries": 0,
                "query_efficiency_success_only": float("nan"),
                "eps": float(args.eps),
                "pgd_steps": int(args.pgd_steps),
                "pgd_step_size": float(args.pgd_step_size),
                "pgd_restarts": int(args.pgd_restarts),
                "pgd_random_start": bool(pgd_random_start),
                "surrogate_epochs": int(args.sur_epochs),
                "surrogate_train_samples": int(len(train_rows)),
            }
        )

    aggregate_and_write(eval_module, csv_dir, "adv_fused_summary", all_tables["adv_fused_summary"], ["name", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_fused_per_class", all_tables["adv_fused_per_class"], ["head", "object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_expert_per_class", all_tables["adv_expert_per_class"], ["head", "object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_expert_specialist", all_tables["adv_expert_specialist"], ["expert", "subset", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_per_class", all_tables["adv_router_per_class"], ["object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_condition", all_tables["adv_router_condition"], ["task", "subset", "attack", "split"])

    eval_module.write_csv(csv_dir / "robustness_summary_aggregate.csv", eval_module.summarize_table(robustness_rows, key_cols=["attack", "subset"]))
    eval_module.write_csv(csv_dir / "metric_degradation_aggregate.csv", eval_module.summarize_table(degradation_rows, key_cols=["attack", "component", "metric"]))
    eval_module.write_csv(csv_dir / "transfer_summary_aggregate.csv", eval_module.summarize_table(transfer_rows, key_cols=["attack"]))

    shared.save_json(
        {
            "attack": attack_name,
            "attack_family": "transfer_based",
            "eps": float(args.eps),
            "pgd_steps": int(args.pgd_steps),
            "pgd_step_size": float(args.pgd_step_size),
            "pgd_restarts": int(args.pgd_restarts),
            "pgd_random_start": bool(pgd_random_start),
            "surrogate_arch": "SurrogateRouterSmall",
            "surrogate_label_definition": "target clean router top-1 pseudo labels",
            "surrogate_epochs": int(args.sur_epochs),
            "surrogate_train_samples": int(len(train_rows)),
            "num_samples": int(len(eval_rows)),
            "run_dirs": [str(p) for p in run_dirs],
            "checkpoint_name": args.checkpoint_name,
            "target_gradient_queries": 0,
            "target_score_queries": 0,
        },
        output_dir / "transfer_attack_manifest.json",
    )

    print(f"[{timestamp()}] Transfer-based attack evaluation complete.")
    print(f"[{timestamp()}] CSV files saved to: {csv_dir}")
    print(f"[{timestamp()}] Margin/drop plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
