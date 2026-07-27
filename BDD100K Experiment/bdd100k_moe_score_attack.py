#!/usr/bin/env python3
"""
BDD100K MoE router robustness evaluation under score-based attacks.

The attack is applied to the router/gating network only, using the clean router
top-1 expert as the reference label because BDD100K has no ground-truth expert
label. Supported score-based attacks are Square and NES, adapted from the GTSRB
score-based gate attack script.

Outputs mirror bdd100k_moe_decision_attack.py and write aggregate CSVs
for adversarial metrics, robustness, degradation, and query summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DECISION_SCRIPT = Path(__file__).resolve().with_name("bdd100k_moe_decision_attack.py")
DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_TEST_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_test.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/data/metadata_files/metadata-New"
DEFAULT_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/New-train-models/moe_stage3"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Evaluation/BDD100k/score_attack_outputs/Square"

EXPERT_NAMES = ["weather", "scene", "time"]
EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}
NUM_ROUTER_CLASSES = 3


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def install_optional_foolbox_stub() -> None:
    """Let us import shared decision-attack helpers even when Foolbox is absent."""
    fb = types.ModuleType("foolbox")
    fb.__file__ = "<score_attack_foolbox_stub>"
    fb.__package__ = "foolbox"
    fb.__spec__ = importlib.util.spec_from_loader("foolbox", loader=None)

    class _UnavailablePyTorchModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Foolbox is unavailable; decision-based attacks cannot run.")

    class _UnavailableAttack:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Foolbox is unavailable; decision-based attacks cannot run.")

    attacks = types.SimpleNamespace(
        HopSkipJumpAttack=_UnavailableAttack,
        BoundaryAttack=_UnavailableAttack,
    )
    fb.PyTorchModel = _UnavailablePyTorchModel
    fb.attacks = attacks
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
    return import_module_from_path("bdd100k_moe_decision_attack_shared", path.resolve())


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    return clamp_01(torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps))


def router_logits_margin_loss(router_logits: torch.Tensor, y_ref: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """Untargeted score loss: maximize best non-reference logit minus reference logit."""
    n = router_logits.size(0)
    true = router_logits[torch.arange(n, device=router_logits.device), y_ref]
    other_logits = router_logits.clone()
    other_logits[torch.arange(n, device=router_logits.device), y_ref] = -1e9
    other = other_logits.max(dim=1).values
    return other - true + kappa


def router_topk_drop_loss(router_logits: torch.Tensor, y_ref: torch.Tensor, topk: int, kappa: float = 0.0) -> torch.Tensor:
    """Maximize kth-other minus reference, pushing the clean route out of top-k."""
    if topk <= 1:
        return router_logits_margin_loss(router_logits, y_ref, kappa=kappa)
    topk = min(topk, NUM_ROUTER_CLASSES - 1)
    n = router_logits.size(0)
    true = router_logits[torch.arange(n, device=router_logits.device), y_ref]
    other_logits = router_logits.clone()
    other_logits[torch.arange(n, device=router_logits.device), y_ref] = -1e9
    kth_other = other_logits.topk(k=topk, dim=1).values[:, -1]
    return kth_other - true + kappa


def router_loss(
    router_logits: torch.Tensor,
    y_ref: torch.Tensor,
    loss_type: str,
    topk: int,
    kappa: float,
) -> torch.Tensor:
    if loss_type == "margin":
        return router_logits_margin_loss(router_logits, y_ref, kappa=kappa)
    if loss_type == "topk_drop":
        return router_topk_drop_loss(router_logits, y_ref, topk=topk, kappa=kappa)
    raise ValueError(f"Unknown loss_type: {loss_type}")


@dataclass
class AttackResult:
    x_adv: torch.Tensor
    queries_per_sample: torch.Tensor
    success: torch.Tensor


@torch.no_grad()
def square_attack_router(
    router: nn.Module,
    x: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float,
    queries: int,
    p_init: float,
    loss_type: str,
    topk: int,
    kappa: float,
    restarts: int,
    early_stop: bool,
    use_uniform_delta: bool,
    min_p: float,
) -> AttackResult:
    device = x.device
    n, c, h, w = x.shape
    x_orig = x.detach().clone()
    base_pred = torch.argmax(router(x_orig), dim=1)

    best_x = x_orig.clone()
    best_loss = torch.full((n,), -1e9, device=device)
    best_success = torch.zeros((n,), dtype=torch.bool, device=device)
    best_queries = torch.full((n,), float(queries), device=device)

    for _restart in range(max(1, restarts)):
        x_adv = project_linf(x_orig + (2.0 * torch.rand_like(x_orig) - 1.0) * eps, x_orig, eps)
        q = torch.zeros((n,), dtype=torch.float32, device=device)

        logits = router(x_adv)
        loss = router_loss(logits, y_ref, loss_type=loss_type, topk=topk, kappa=kappa)
        q += 1.0
        active = torch.argmax(logits, dim=1).eq(base_pred) if early_stop else torch.ones((n,), dtype=torch.bool, device=device)

        for step in range(1, queries):
            if early_stop and not bool(active.any()):
                break

            frac = step / max(1, queries - 1)
            p = max(min_p, p_init * (1.0 - 0.9 * frac))
            side = max(1, int(round(min(h, w) * p)))
            if step % 10 == 0:
                side = max(1, side // 2)

            x_cand = x_adv.clone()
            y0 = torch.randint(0, h - side + 1, (n,), device=device)
            x0 = torch.randint(0, w - side + 1, (n,), device=device)

            for i in torch.where(active)[0].tolist():
                yy = int(y0[i])
                xx = int(x0[i])
                if use_uniform_delta:
                    delta = torch.empty((c, side, side), device=device).uniform_(-eps, eps)
                else:
                    delta = (torch.randint(0, 2, (c, side, side), device=device, dtype=torch.float32) * 2.0 - 1.0) * eps
                x_cand[i, :, yy : yy + side, xx : xx + side] = clamp_01(
                    x_orig[i, :, yy : yy + side, xx : xx + side] + delta
                )

            x_cand = project_linf(x_cand, x_orig, eps)
            logits_c = router(x_cand)
            loss_c = router_loss(logits_c, y_ref, loss_type=loss_type, topk=topk, kappa=kappa)
            q[active] += 1.0

            improve = (loss_c > loss) & active
            if bool(improve.any()):
                x_adv[improve] = x_cand[improve]
                loss[improve] = loss_c[improve]

            if early_stop:
                active = torch.argmax(router(x_adv), dim=1).eq(base_pred)

        success = torch.argmax(router(x_adv), dim=1).ne(base_pred)
        select = (success & ~best_success) | (success & best_success & (loss > best_loss)) | (~success & ~best_success & (loss > best_loss))
        if bool(select.any()):
            best_x[select] = x_adv[select]
            best_loss[select] = loss[select]
            best_success[select] = success[select]
            best_queries[select] = q[select].clamp(min=1.0, max=float(queries))

    final_success = torch.argmax(router(best_x), dim=1).ne(base_pred)
    return AttackResult(x_adv=best_x, queries_per_sample=best_queries, success=final_success)


@torch.no_grad()
def nes_attack_router(
    router: nn.Module,
    x: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float,
    queries: int,
    lr: float,
    sigma: float,
    samples_per_step: int,
    loss_type: str,
    topk: int,
    kappa: float,
    restarts: int,
    early_stop: bool,
    momentum: float,
    grad_norm: bool,
) -> AttackResult:
    device = x.device
    x_orig = x.detach().clone()
    base_pred = torch.argmax(router(x_orig), dim=1)
    n = x.size(0)

    best_x = x_orig.clone()
    best_loss = torch.full((n,), -1e9, device=device)
    best_success = torch.zeros((n,), dtype=torch.bool, device=device)
    best_queries = torch.full((n,), float(queries), device=device)

    for _restart in range(max(1, restarts)):
        x_adv = x_orig.clone()
        q = torch.zeros((n,), dtype=torch.float32, device=device)
        mom = torch.zeros_like(x_adv)
        active = torch.ones((n,), dtype=torch.bool, device=device)

        while q.max().item() + 2 * samples_per_step <= queries:
            if early_stop and not bool(active.any()):
                break

            grad = torch.zeros_like(x_adv)
            for _sample in range(samples_per_step):
                u = torch.randn_like(x_adv)
                x_plus = project_linf(x_adv + sigma * u, x_orig, eps)
                x_minus = project_linf(x_adv - sigma * u, x_orig, eps)

                l_plus = router_loss(router(x_plus), y_ref, loss_type=loss_type, topk=topk, kappa=kappa)
                l_minus = router_loss(router(x_minus), y_ref, loss_type=loss_type, topk=topk, kappa=kappa)
                grad += (l_plus - l_minus).view(-1, 1, 1, 1) * u

            grad = grad / (2.0 * sigma * samples_per_step)
            if grad_norm:
                flat = grad.view(n, -1)
                grad = (flat / (flat.norm(dim=1, keepdim=True) + 1e-12)).view_as(grad)

            if momentum > 0.0:
                mom = momentum * mom + (1.0 - momentum) * grad
                step_dir = torch.sign(mom)
            else:
                step_dir = torch.sign(grad)

            x_adv = project_linf(x_adv + lr * step_dir, x_orig, eps)
            q[active] += float(2 * samples_per_step)

            if early_stop:
                active = torch.argmax(router(x_adv), dim=1).eq(base_pred)

        loss = router_loss(router(x_adv), y_ref, loss_type=loss_type, topk=topk, kappa=kappa)
        success = torch.argmax(router(x_adv), dim=1).ne(base_pred)
        q = q.clamp(min=1.0, max=float(queries))

        select = (success & ~best_success) | (success & best_success & (loss > best_loss)) | (~success & ~best_success & (loss > best_loss))
        if bool(select.any()):
            best_x[select] = x_adv[select]
            best_loss[select] = loss[select]
            best_success[select] = success[select]
            best_queries[select] = q[select]

    final_success = torch.argmax(router(best_x), dim=1).ne(base_pred)
    return AttackResult(x_adv=best_x, queries_per_sample=best_queries, success=final_success)


def craft_adversarial_images_score(
    router: nn.Module,
    loader,
    clean_router_labels: torch.Tensor,
    device: torch.device,
    attack_name: str,
    eps: float,
    queries: int,
    loss_type: str,
    topk_loss: int,
    kappa: float,
    restarts: int,
    early_stop: bool,
    square_p_init: float,
    square_min_p: float,
    square_uniform_delta: bool,
    nes_lr: float,
    nes_sigma: float,
    nes_samples_per_step: int,
    nes_momentum: float,
    nes_grad_norm: bool,
    progress_label: str,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    adv_chunks: List[torch.Tensor] = []
    success_chunks: List[np.ndarray] = []
    query_chunks: List[np.ndarray] = []
    offset = 0
    total_batches = len(loader)

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

        if attack_name == "square":
            result = square_attack_router(
                router=router,
                x=xb,
                y_ref=yb,
                eps=eps,
                queries=queries,
                p_init=square_p_init,
                loss_type=loss_type,
                topk=topk_loss,
                kappa=kappa,
                restarts=restarts,
                early_stop=early_stop,
                use_uniform_delta=square_uniform_delta,
                min_p=square_min_p,
            )
        elif attack_name == "nes":
            result = nes_attack_router(
                router=router,
                x=xb,
                y_ref=yb,
                eps=eps,
                queries=queries,
                lr=nes_lr,
                sigma=nes_sigma,
                samples_per_step=nes_samples_per_step,
                loss_type=loss_type,
                topk=topk_loss,
                kappa=kappa,
                restarts=restarts,
                early_stop=early_stop,
                momentum=nes_momentum,
                grad_norm=nes_grad_norm,
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        adv_chunks.append(result.x_adv.detach().cpu())
        success_chunks.append(result.success.detach().cpu().numpy().astype(np.bool_).reshape(-1))
        query_chunks.append(result.queries_per_sample.detach().cpu().numpy().astype(np.float32).reshape(-1))

    return (
        torch.cat(adv_chunks, dim=0),
        np.concatenate(success_chunks, axis=0),
        np.concatenate(query_chunks, axis=0),
    )


def aggregate_and_write(eval_module, csv_dir: Path, prefix: str, rows: List[Dict[str, Any]], key_cols: Sequence[str]) -> None:
    eval_module.write_csv(csv_dir / f"{prefix}_aggregate.csv", eval_module.summarize_table(rows, key_cols=key_cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="BDD100K MoE score-based router attack evaluation")
    parser.add_argument("--decision-script", type=str, default=str(DEFAULT_DECISION_SCRIPT))
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--test-script", type=str, default=DEFAULT_TEST_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--run-glob", type=str, default=DEFAULT_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack", choices=["square", "Square", "NES", "nes"], default="square")
    parser.add_argument("--eps", type=float, default=0.09, help="L-infinity budget in pixel space [0, 1].")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["margin", "topk_drop"], default="margin")
    parser.add_argument("--topk-loss", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--early-stop", action="store_true", help="Stop each sample once the router flips.")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--square-p-init", type=float, default=0.25)
    parser.add_argument("--square-min-p", type=float, default=0.01)
    parser.add_argument("--square-uniform-delta", action="store_true")
    parser.add_argument("--nes-lr", type=float, default=0.02)
    parser.add_argument("--nes-sigma", type=float, default=0.001)
    parser.add_argument("--nes-samples-per-step", type=int, default=20)
    parser.add_argument("--nes-momentum", type=float, default=0.9)
    parser.add_argument("--no-nes-grad-norm", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    shared = import_decision_helpers(Path(args.decision_script))
    shared.ensure_train_import_dependencies()
    train_module = shared.import_module_from_path("bdd100k_moe_train_score_attack_module", Path(args.train_script).resolve())
    eval_module = shared.import_module_from_path("bdd100k_moe_test_score_attack_module", Path(args.test_script).resolve())

    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    ensure_dir(csv_dir)
    ensure_dir(plots_dir)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    attack_name = args.attack.lower()
    early_stop = bool(args.early_stop) and not args.no_early_stop

    bundle = shared.load_json(metadata_dir / "metadata_bundle.json")
    rows = shared.load_json(metadata_dir / "test_fixed.json")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    run_dirs = shared.list_run_dirs(args.run_dirs, args.run_glob)

    print(f"[{timestamp()}] Evaluating {len(run_dirs)} run(s) on {len(rows)} image(s).")
    print(f"[{timestamp()}] Attack={attack_name}, eps={args.eps}, queries={args.queries}, loss={args.loss_type}")

    all_tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    robustness_rows: List[Dict[str, Any]] = []
    degradation_rows: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []
    query_summary_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        print(f"[{timestamp()}] Run: {run_dir}", flush=True)
        model, config = shared.load_model_for_run(run_dir, args.checkpoint_name, train_module, bundle, device)
        normalized_moe = shared.NormalizedMoE(model).to(device).eval()
        router = shared.RouterOnly(normalized_moe).to(device).eval()

        image_size = int(config.get("image_size", 224))
        pixel_dataset = shared.BDD100KPixelDataset(rows, image_size=image_size)
        clean_loader = shared.make_loader(pixel_dataset, args.batch_size, args.num_workers, device)

        clean_router_labels = shared.collect_clean_router_labels(
            router,
            clean_loader,
            device,
            progress_label=f"{run_dir.name} clean router labels",
        )
        adv_images, success, queries_per_sample = craft_adversarial_images_score(
            router=router,
            loader=clean_loader,
            clean_router_labels=clean_router_labels,
            device=device,
            attack_name=attack_name,
            eps=float(args.eps),
            queries=int(args.queries),
            loss_type=args.loss_type,
            topk_loss=int(args.topk_loss),
            kappa=float(args.kappa),
            restarts=int(args.restarts),
            early_stop=early_stop,
            square_p_init=float(args.square_p_init),
            square_min_p=float(args.square_min_p),
            square_uniform_delta=bool(args.square_uniform_delta),
            nes_lr=float(args.nes_lr),
            nes_sigma=float(args.nes_sigma),
            nes_samples_per_step=int(args.nes_samples_per_step),
            nes_momentum=float(args.nes_momentum),
            nes_grad_norm=not args.no_nes_grad_norm,
            progress_label=f"{run_dir.name} {attack_name} score attack",
        )

        adv_dataset = shared.DictTensorDataset(adv_images, pixel_dataset)
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

        seed = int(config.get("seed", -1))
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

        query_summary_rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack_name,
                "n": int(queries_per_sample.size),
                "success_rate": float(np.mean(success)) if success.size else float("nan"),
                "queries_mean": float(np.mean(queries_per_sample)) if queries_per_sample.size else float("nan"),
                "queries_median": float(np.median(queries_per_sample)) if queries_per_sample.size else float("nan"),
                "queries_std": float(np.std(queries_per_sample)) if queries_per_sample.size else float("nan"),
                "queries_min": float(np.min(queries_per_sample)) if queries_per_sample.size else float("nan"),
                "queries_max": float(np.max(queries_per_sample)) if queries_per_sample.size else float("nan"),
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

    shared.save_json(
        {
            "attack": attack_name,
            "attack_family": "score_based",
            "eps": float(args.eps),
            "queries": int(args.queries),
            "loss_type": args.loss_type,
            "topk_loss": int(args.topk_loss),
            "kappa": float(args.kappa),
            "restarts": int(args.restarts),
            "early_stop": early_stop,
            "num_samples": len(rows),
            "run_dirs": [str(p) for p in run_dirs],
            "checkpoint_name": args.checkpoint_name,
            "router_label_definition": "clean_router_top1",
            "square": {
                "p_init": float(args.square_p_init),
                "min_p": float(args.square_min_p),
                "uniform_delta": bool(args.square_uniform_delta),
            },
            "nes": {
                "lr": float(args.nes_lr),
                "sigma": float(args.nes_sigma),
                "samples_per_step": int(args.nes_samples_per_step),
                "momentum": float(args.nes_momentum),
                "grad_norm": not args.no_nes_grad_norm,
            },
        },
        output_dir / "score_attack_manifest.json",
    )

    print(f"[{timestamp()}] Score-based attack evaluation complete.")
    print(f"[{timestamp()}] CSV files saved to: {csv_dir}")
    print(f"[{timestamp()}] Margin/drop plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
