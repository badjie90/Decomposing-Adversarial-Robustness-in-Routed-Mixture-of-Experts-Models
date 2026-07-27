#!/usr/bin/env python3
"""
BDD100K MoE router robustness evaluation under decision-based attacks.

The attack is applied to the router/gating network only. Because BDD100K has no
ground-truth expert label, the hard-label objective uses the clean router top-1
expert as the label and measures whether the attack flips that routing decision.

Outputs mirror the normal BDD100K evaluation CSVs for both clean and attacked
inputs, then add robustness CSVs that compare router behavior, individual expert
heads, and fused MoE output under attack.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
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
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


def install_foolbox_brendel_stub() -> None:
    """Avoid importing Foolbox Brendel-Bethge attacks when numba is broken."""

    def _stub_unavailable_attack(name: str):
        class _UnavailableAttack:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    f"{name} is unavailable in this environment. Use hsj or boundary."
                )
        _UnavailableAttack.__name__ = name
        return _UnavailableAttack

    bb = types.ModuleType("foolbox.attacks.brendel_bethge")
    bb.__file__ = "<foolbox_brendel_bethge_stub>"
    bb.__package__ = "foolbox.attacks"
    bb.__spec__ = importlib.machinery.ModuleSpec("foolbox.attacks.brendel_bethge", loader=None)
    bb.__getattr__ = lambda name: _stub_unavailable_attack(name)
    bb.BrendelBethgeAttack = _stub_unavailable_attack("BrendelBethgeAttack")
    bb.L0BrendelBethgeAttack = _stub_unavailable_attack("L0BrendelBethgeAttack")
    bb.L1BrendelBethgeAttack = _stub_unavailable_attack("L1BrendelBethgeAttack")
    bb.L2BrendelBethgeAttack = _stub_unavailable_attack("L2BrendelBethgeAttack")
    bb.LinfBrendelBethgeAttack = _stub_unavailable_attack("LinfBrendelBethgeAttack")
    bb.LinfinityBrendelBethgeAttack = bb.LinfBrendelBethgeAttack
    sys.modules["foolbox.attacks.brendel_bethge"] = bb


install_foolbox_brendel_stub()

try:
    import foolbox as fb
except ImportError as exc:
    raise ImportError("foolbox is required. Install with: pip install foolbox") from exc

try:
    from torchvision import transforms
except ImportError as exc:
    raise ImportError("torchvision is required. Install with: pip install torchvision") from exc


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

TARGET_OBJECTS = ["car", "pedestrian", "traffic_sign"]
EXPERT_NAMES = ["weather", "scene", "time"]
EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_TEST_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_test.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/data/metadata_files/metadata-New"
DEFAULT_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/New-train-models/moe_stage3"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Evaluation/BDD100k/decision_attack_outputs/HSJ"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_train_import_dependencies() -> None:
    """Allow architecture-only imports from the train script in lean eval envs."""

    try:
        import torchmetrics.classification  # noqa: F401
        return
    except ImportError:
        pass

    class _UnavailableMetric:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchmetrics is unavailable; metric training helpers cannot be used.")

    torchmetrics_mod = types.ModuleType("torchmetrics")
    classification_mod = types.ModuleType("torchmetrics.classification")
    classification_mod.MultilabelAUROC = _UnavailableMetric
    classification_mod.MultilabelF1Score = _UnavailableMetric
    classification_mod.MulticlassAccuracy = _UnavailableMetric
    torchmetrics_mod.classification = classification_mod
    sys.modules.setdefault("torchmetrics", torchmetrics_mod)
    sys.modules.setdefault("torchmetrics.classification", classification_mod)


def list_run_dirs(run_dirs: Sequence[str], run_glob: str) -> List[Path]:
    if run_dirs:
        resolved = [Path(p).expanduser().resolve() for p in run_dirs]
    else:
        glob_path = Path(run_glob).expanduser()
        if glob_path.is_absolute():
            resolved = sorted(glob_path.parent.glob(glob_path.name))
        else:
            resolved = sorted(Path(".").glob(run_glob))
        resolved = [p.resolve() for p in resolved]
    resolved = [p for p in resolved if p.exists()]
    if not resolved:
        raise FileNotFoundError(f"No run directories found. Got --run-glob={run_glob!r}")
    return resolved


class BDD100KPixelDataset(Dataset):
    """BDD100K test rows with unnormalized [0, 1] images for Foolbox bounds."""

    def __init__(self, rows: Sequence[Dict[str, Any]], image_size: int) -> None:
        self.rows = list(rows)
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "weather_id": torch.tensor(row["weather_id"], dtype=torch.long),
            "scene_id": torch.tensor(row["scene_id"], dtype=torch.long),
            "time_id": torch.tensor(row["time_id"], dtype=torch.long),
            "weather_mask": torch.tensor(row["weather_mask"], dtype=torch.float32),
            "scene_mask": torch.tensor(row["scene_mask"], dtype=torch.float32),
            "time_mask": torch.tensor(row["time_mask"], dtype=torch.float32),
            "objects": torch.tensor(
                [row["car_present"], row["pedestrian_present"], row["traffic_sign_present"]],
                dtype=torch.float32,
            ),
            "image_path": row["image_path"],
        }


class NormalizedMoE(nn.Module):
    """Accepts pixel-space tensors, normalizes, then calls the trained MoE."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model((image - self.mean) / self.std)


class RouterOnly(nn.Module):
    def __init__(self, normalized_moe: NormalizedMoE) -> None:
        super().__init__()
        self.normalized_moe = normalized_moe

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.normalized_moe(image)["router_logits"]


class DictTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, source_dataset: Dataset) -> None:
        self.images = images
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.source_dataset[index])
        item["image"] = self.images[index]
        return item


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def numpy_from_list(chunks: List[np.ndarray], ndim: int = 2) -> np.ndarray:
    if not chunks:
        shape = (0,) if ndim == 1 else (0, 0)
        return np.empty(shape, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    progress_label: str = "prediction",
) -> Dict[str, Any]:
    model.eval()
    chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    image_paths: List[str] = []
    batch_ms: List[float] = []
    num_images = 0
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        image_paths.extend(batch["image_path"])
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        print(
            f"[{timestamp()}] {progress_label}: batch {batch_idx}/{total_batches} "
            f"({num_images + int(images.shape[0])}/{len(loader.dataset)} images)",
            flush=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_ms.append((time.perf_counter() - start) * 1000.0)
        num_images += int(images.shape[0])

        chunks["y_true"].append(batch["objects"].detach().cpu().numpy())
        for key in ["weather_id", "scene_id", "time_id", "weather_mask", "scene_mask", "time_mask"]:
            chunks[key].append(batch[key].detach().cpu().numpy())

        chunks["probs_fused"].append(torch.sigmoid(outputs["obj_fused_logits"]).detach().cpu().numpy())
        chunks["probs_weather"].append(torch.sigmoid(outputs["obj_w_logits"]).detach().cpu().numpy())
        chunks["probs_scene"].append(torch.sigmoid(outputs["obj_s_logits"]).detach().cpu().numpy())
        chunks["probs_time"].append(torch.sigmoid(outputs["obj_t_logits"]).detach().cpu().numpy())
        chunks["weather_logits"].append(outputs["weather_logits"].detach().cpu().numpy())
        chunks["scene_logits"].append(outputs["scene_logits"].detach().cpu().numpy())
        chunks["time_logits"].append(outputs["time_logits"].detach().cpu().numpy())
        chunks["alpha"].append(outputs["alpha"].detach().cpu().numpy())
        chunks["router_logits"].append(outputs["router_logits"].detach().cpu().numpy())

    return {
        "image_path": image_paths,
        "y_true": numpy_from_list(chunks["y_true"]),
        "weather_id": numpy_from_list(chunks["weather_id"], ndim=1),
        "scene_id": numpy_from_list(chunks["scene_id"], ndim=1),
        "time_id": numpy_from_list(chunks["time_id"], ndim=1),
        "weather_mask": numpy_from_list(chunks["weather_mask"], ndim=1),
        "scene_mask": numpy_from_list(chunks["scene_mask"], ndim=1),
        "time_mask": numpy_from_list(chunks["time_mask"], ndim=1),
        "probs_fused": numpy_from_list(chunks["probs_fused"]),
        "probs_weather": numpy_from_list(chunks["probs_weather"]),
        "probs_scene": numpy_from_list(chunks["probs_scene"]),
        "probs_time": numpy_from_list(chunks["probs_time"]),
        "weather_logits": numpy_from_list(chunks["weather_logits"]),
        "scene_logits": numpy_from_list(chunks["scene_logits"]),
        "time_logits": numpy_from_list(chunks["time_logits"]),
        "alpha": numpy_from_list(chunks["alpha"]),
        "router_logits": numpy_from_list(chunks["router_logits"]),
        "inference_ms_per_image": float(np.sum(batch_ms) / max(num_images, 1)),
    }


def configure_foolbox_imports() -> None:
    """Match the GTSRB script's workaround for unavailable Brendel-Bethge deps."""
    install_foolbox_brendel_stub()


def build_attacker(name: str, steps: int):
    key = name.lower()
    if key == "hsj":
        return fb.attacks.HopSkipJumpAttack(
            steps=steps,
            initial_gradient_eval_steps=200,
            max_gradient_eval_steps=10000,
            gamma=0.5,
        )
    if key == "boundary":
        return fb.attacks.BoundaryAttack(
            steps=steps,
            spherical_step=0.005,
            source_step=0.005,
        )
    raise ValueError(f"Unknown attack: {name}")


@torch.no_grad()
def router_margin_loss_np(router: nn.Module, x: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    logits = router(x)
    n = logits.size(0)
    true = logits[torch.arange(n, device=logits.device), y_ref]
    other_logits = logits.clone()
    other_logits[torch.arange(n, device=logits.device), y_ref] = -1e9
    other = other_logits.max(dim=1).values
    return other - true


@torch.no_grad()
def amplify_router_adversarials(
    router: nn.Module,
    x: torch.Tensor,
    x_adv: torch.Tensor,
    y_ref: torch.Tensor,
    eps: Optional[float],
) -> Tuple[torch.Tensor, np.ndarray]:
    """Use more of the L-infinity budget while preserving router misrouting."""
    if eps is None or eps <= 0:
        success = torch.argmax(router(x_adv), dim=1).ne(y_ref)
        return x_adv, success.detach().cpu().numpy().astype(np.bool_)

    best_x = x_adv.detach().clone()
    best_loss = router_margin_loss_np(router, best_x, y_ref)
    best_success = torch.argmax(router(best_x), dim=1).ne(y_ref)
    direction = best_x - x
    candidates = []
    for scale in (1.25, 1.5, 2.0, 3.0, 4.0):
        candidates.append(torch.clamp(torch.max(torch.min(x + scale * direction, x + eps), x - eps), 0.0, 1.0))
    candidates.append(torch.clamp(torch.max(torch.min(x + eps * torch.sign(direction), x + eps), x - eps), 0.0, 1.0))

    for cand in candidates:
        logits = router(cand)
        success = torch.argmax(logits, dim=1).ne(y_ref)
        loss = router_margin_loss_np(router, cand, y_ref)
        select = success & ((~best_success) | (loss > best_loss))
        if bool(select.any()):
            best_x[select] = cand[select]
            best_loss[select] = loss[select]
            best_success[select] = success[select]

    return best_x, best_success.detach().cpu().numpy().astype(np.bool_)


def load_model_for_run(
    run_dir: Path,
    checkpoint_name: str,
    train_module,
    metadata_bundle: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    config = load_json(run_dir / "config.json")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    weather_names = [k for k, _ in sorted(metadata_bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(metadata_bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(metadata_bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    stage = str(config.get("stage", "stage3_moe"))
    if stage == "stage0_baseline":
        raise ValueError("Decision attack requires a MoE checkpoint with a router; stage0_baseline has no router.")

    model = train_module.build_model(
        stage=stage,
        backbone_name=str(config.get("backbone_name", "convnextv2_tiny.fcmae_ft_in1k")),
        num_weather=len(weather_names),
        num_scene=len(scene_names),
        num_time=len(time_names),
        pretrained=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config


@torch.no_grad()
def collect_clean_router_labels(
    router: nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_label: str = "clean router labels",
) -> torch.Tensor:
    labels: List[torch.Tensor] = []
    router.eval()
    total_batches = len(loader)
    num_images = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        num_images += int(images.shape[0])
        print(
            f"[{timestamp()}] {progress_label}: batch {batch_idx}/{total_batches} "
            f"({num_images}/{len(loader.dataset)} images)",
            flush=True,
        )
        labels.append(torch.argmax(router(images), dim=1).detach().cpu())
    return torch.cat(labels, dim=0).long()


def craft_adversarial_images(
    router: nn.Module,
    loader: DataLoader,
    clean_router_labels: torch.Tensor,
    device: torch.device,
    attack_name: str,
    eps: Optional[float],
    steps: int,
    progress_label: str = "adversarial attack",
) -> Tuple[torch.Tensor, np.ndarray]:
    fmodel = fb.PyTorchModel(router, bounds=(0.0, 1.0), device=device)
    attacker = build_attacker(attack_name, steps)
    adv_chunks: List[torch.Tensor] = []
    success_chunks: List[np.ndarray] = []
    offset = 0

    def run_attack_safely(x: torch.Tensor, y: torch.Tensor, attack_kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray]:
        try:
            _raw, clipped, success = attacker(fmodel, x, y, **attack_kwargs)
            try:
                success_np = success.detach().cpu().numpy().astype(np.bool_)
            except Exception:
                success_np = np.asarray(success, dtype=np.bool_)
            return clipped.detach(), success_np.reshape(-1)
        except ValueError as exc:
            if "init_attack failed" not in str(exc):
                raise

        # HSJ can fail to find an initial adversarial point for only part of a
        # batch. Retry one image at a time and keep clean images for failures.
        clipped_batch = x.detach().clone()
        success_batch = np.zeros((x.shape[0],), dtype=np.bool_)
        failed = 0
        for i in range(x.shape[0]):
            try:
                _raw_i, clipped_i, success_i = attacker(fmodel, x[i : i + 1], y[i : i + 1], **attack_kwargs)
                clipped_batch[i : i + 1] = clipped_i.detach()
                try:
                    success_i_np = success_i.detach().cpu().numpy().astype(np.bool_)
                except Exception:
                    success_i_np = np.asarray(success_i, dtype=np.bool_)
                success_batch[i] = bool(success_i_np.reshape(-1)[0])
            except ValueError as exc:
                if "init_attack failed" not in str(exc):
                    raise
                failed += 1
        if failed:
            print(
                f"[{timestamp()}] Warning: {attack_name} init_attack failed for "
                f"{failed}/{x.shape[0]} sample(s); using clean images for those samples.",
                flush=True,
            )
        return clipped_batch, success_batch

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

        attack_kwargs = {} if eps is None else {"epsilons": eps}
        clipped, success_np = run_attack_safely(xb, yb, attack_kwargs)
        clipped, success_np = amplify_router_adversarials(router, xb, clipped, yb, eps)
        adv_chunks.append(clipped.detach().cpu())
        success_chunks.append(success_np.reshape(-1))

    return torch.cat(adv_chunks, dim=0), np.concatenate(success_chunks, axis=0)


def add_run_info(rows: List[Dict[str, Any]], seed: int, run_dir: Path, attack: str, split: str) -> None:
    for row in rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)
        row["attack"] = attack
        row["split"] = split


def compute_eval_tables(
    preds: Dict[str, Any],
    eval_module,
    weather_names: Sequence[str],
    scene_names: Sequence[str],
    time_names: Sequence[str],
    threshold: float,
    seed: int,
    run_dir: Path,
    attack: str,
    split: str,
) -> Dict[str, List[Dict[str, Any]]]:
    fused_summary = eval_module.compute_multilabel_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    fused_summary["inference_ms"] = preds["inference_ms_per_image"]
    fused_rows = [fused_summary]

    fused_per_class = eval_module.compute_per_class_object_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    expert_per_class: List[Dict[str, Any]] = []
    for head_name, probs in [
        ("weather_expert", preds["probs_weather"]),
        ("scene_expert", preds["probs_scene"]),
        ("time_expert", preds["probs_time"]),
    ]:
        expert_per_class.extend(eval_module.compute_per_class_object_metrics(head_name, preds["y_true"], probs, threshold))

    specialist = eval_module.compute_specialist_metrics(preds, weather_names, scene_names, time_names, threshold)

    condition: List[Dict[str, Any]] = []
    for task, logits, y_true, mask, names in [
        ("weather", preds["weather_logits"], preds["weather_id"], preds["weather_mask"], weather_names),
        ("scene", preds["scene_logits"], preds["scene_id"], preds["scene_mask"], scene_names),
        ("time", preds["time_logits"], preds["time_id"], preds["time_mask"], time_names),
    ]:
        row, _ = eval_module.compute_condition_metrics(task, logits, y_true, mask, names)
        condition.append(row)

    router_summary, _ = eval_module.compute_router_metrics(preds, weather_names, scene_names, time_names)
    router_per_class = eval_module.compute_router_per_class_rows(preds)
    router_condition = eval_module.compute_router_condition_rows(preds, weather_names, scene_names, time_names)

    tables = {
        "fused_summary": fused_rows,
        "fused_per_class": fused_per_class,
        "expert_per_class": expert_per_class,
        "expert_specialist": specialist,
        "condition": condition,
        "router_summary": router_summary,
        "router_per_class": router_per_class,
        "router_condition": router_condition,
    }
    for rows in tables.values():
        add_run_info(rows, seed, run_dir, attack, split)
    return tables


def routing_flip_rows(clean: Dict[str, Any], adv: Dict[str, Any], success: np.ndarray, seed: int, run_dir: Path, attack: str) -> List[Dict[str, Any]]:
    clean_top1 = np.argmax(clean["alpha"], axis=1)
    adv_top1 = np.argmax(adv["alpha"], axis=1)
    rows = [
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "subset": "all",
            "n": int(clean_top1.size),
            "attack_success_rate_router": float(np.mean(success)) if success.size else float("nan"),
            "router_top1_flip_rate": float(np.mean(clean_top1 != adv_top1)),
            "router_mean_l1_alpha_shift": float(np.mean(np.abs(clean["alpha"] - adv["alpha"]).sum(axis=1))),
            "router_mean_l2_alpha_shift": float(np.mean(np.linalg.norm(clean["alpha"] - adv["alpha"], axis=1))),
            "fused_mean_abs_prob_shift": float(np.mean(np.abs(clean["probs_fused"] - adv["probs_fused"]))),
        }
    ]
    for i, name in enumerate(EXPERT_NAMES):
        mask = clean_top1 == i
        if not np.any(mask):
            continue
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack,
                "subset": f"clean_top1_{name}",
                "n": int(mask.sum()),
                "attack_success_rate_router": float(np.mean(success[mask])),
                "router_top1_flip_rate": float(np.mean(clean_top1[mask] != adv_top1[mask])),
                "router_mean_l1_alpha_shift": float(np.mean(np.abs(clean["alpha"][mask] - adv["alpha"][mask]).sum(axis=1))),
                "router_mean_l2_alpha_shift": float(np.mean(np.linalg.norm(clean["alpha"][mask] - adv["alpha"][mask], axis=1))),
                "fused_mean_abs_prob_shift": float(np.mean(np.abs(clean["probs_fused"][mask] - adv["probs_fused"][mask]))),
            }
        )
    return rows


def degradation_rows(clean_tables: Dict[str, List[Dict[str, Any]]], adv_tables: Dict[str, List[Dict[str, Any]]], seed: int, run_dir: Path, attack: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_names = ["micro_f1", "macro_f1", "accuracy", "balanced_accuracy", "mAP", "micro_mAP", "macro_auroc", "micro_auroc", "ece", "brier"]
    clean_lookup = {row["name"]: row for row in clean_tables["fused_summary"]}
    adv_lookup = {row["name"]: row for row in adv_tables["fused_summary"]}
    for name, c_row in clean_lookup.items():
        a_row = adv_lookup.get(name)
        if a_row is None:
            continue
        for metric in metric_names:
            if metric in c_row and metric in a_row:
                rows.append(
                    {
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "attack": attack,
                        "component": name,
                        "metric": metric,
                        "clean": c_row[metric],
                        "adv": a_row[metric],
                        "delta_adv_minus_clean": float(a_row[metric]) - float(c_row[metric]),
                    }
                )
    return rows


def router_reference_margins(alpha: np.ndarray, reference_top1: np.ndarray) -> np.ndarray:
    margins = np.empty((alpha.shape[0],), dtype=np.float64)
    for i, ref in enumerate(reference_top1.astype(np.int64)):
        other = np.delete(alpha[i], ref)
        margins[i] = float(alpha[i, ref] - np.max(other))
    return margins


def positive_object_fused_probability_drops(
    clean_probs: np.ndarray,
    adv_probs: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    all_drops: List[np.ndarray] = []
    by_object: Dict[str, np.ndarray] = {}
    for obj_idx, obj_name in enumerate(TARGET_OBJECTS):
        mask = y_true[:, obj_idx] == 1
        drops = clean_probs[mask, obj_idx] - adv_probs[mask, obj_idx]
        drops = drops.astype(np.float64)
        by_object[obj_name] = drops
        if drops.size:
            all_drops.append(drops)
    if all_drops:
        return np.concatenate(all_drops, axis=0), by_object
    return np.empty((0,), dtype=np.float64), by_object


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def plot_hist(path: Path, values: np.ndarray, title: str, xlabel: str, color: str) -> None:
    values = finite_values(values)
    if values.size == 0:
        return
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(values, bins=50, color=color, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=21, fontweight="bold", labelpad=18)
    ax.set_ylabel("Number of input samples", fontsize=21, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")
        tick_label.set_fontweight("bold")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def margin_plot_rows_and_files(
    clean: Dict[str, Any],
    adv: Dict[str, Any],
    plots_dir: Path,
    seed: int,
    run_dir: Path,
    attack: str,
) -> List[Dict[str, Any]]:
    run_plot_dir = plots_dir / f"seed_{seed if seed >= 0 else run_dir.name}"
    clean_top1 = np.argmax(clean["alpha"], axis=1)

    router_margin_clean = router_reference_margins(clean["alpha"], clean_top1)
    router_margin_adv = router_reference_margins(adv["alpha"], clean_top1)
    router_margin_degradation = router_margin_clean - router_margin_adv

    fused_drop_all, fused_drop_by_object = positive_object_fused_probability_drops(
        clean["probs_fused"],
        adv["probs_fused"],
        clean["y_true"],
    )

    plot_hist(
        run_plot_dir / f"gate_margin_clean_{attack}.png",
        router_margin_clean,
        f"Gate Margin (Clean Baseline)",
        "Clean Gate margin",
        color="green",
    )
    plot_hist(
        run_plot_dir / f"gate_margin_adv_{attack}.png",
        router_margin_adv,
        f"Gate Margin Adv ({attack})",
        "Gate margin (Adversarial)",
        color="orange",
    )
    plot_hist(
        run_plot_dir / f"gate_margin_degradation_{attack}.png",
        router_margin_degradation,
        f"Gate Margin Degradation ({attack})",
        "Gate margin drop after attack",
        color="crimson",
    )
    plot_hist(
        run_plot_dir / f"fused_positive_object_prob_drop_{attack}.png",
        fused_drop_all,
        f"FPoPD ({attack})",
        "Object probability drop after attack",
        color="crimson",
    )
    for obj_name, drops in fused_drop_by_object.items():
        plot_hist(
            run_plot_dir / f"fused_positive_object_prob_drop_{obj_name}_{attack}.png",
            drops,
            f"FPoPD - {obj_name} ({attack})",
            "Object probability drop after attack",
            color="crimson",
        )

    rows = [
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "name": "router_margin_clean",
            "n": int(router_margin_clean.size),
            "mean": float(np.mean(router_margin_clean)),
            "median": float(np.median(router_margin_clean)),
            "std": float(np.std(router_margin_clean)),
        },
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "name": "router_margin_adv",
            "n": int(router_margin_adv.size),
            "mean": float(np.mean(router_margin_adv)),
            "median": float(np.median(router_margin_adv)),
            "std": float(np.std(router_margin_adv)),
        },
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "name": "router_margin_degradation",
            "n": int(router_margin_degradation.size),
            "mean": float(np.mean(router_margin_degradation)),
            "median": float(np.median(router_margin_degradation)),
            "std": float(np.std(router_margin_degradation)),
        },
    ]

    drop_specs = [("fused_positive_object_prob_drop_all", fused_drop_all)]
    drop_specs.extend((f"fused_positive_object_prob_drop_{obj_name}", drops) for obj_name, drops in fused_drop_by_object.items())
    for name, drops in drop_specs:
        drops = finite_values(drops)
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack,
                "name": name,
                "n": int(drops.size),
                "mean": float(np.mean(drops)) if drops.size else float("nan"),
                "median": float(np.median(drops)) if drops.size else float("nan"),
                "std": float(np.std(drops)) if drops.size else float("nan"),
            }
        )
    return rows


def fused_attack_comparison_plots(
    preds: Dict[str, Any],
    eval_module,
    plots_dir: Path,
    seed: int,
    run_dir: Path,
    attack: str,
    plot_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the fused PR, ROC, and reliability plots used by the clean baseline."""
    if plot_cfg is None:
        plot_cfg = {
            "title_fontsize": 25,
            "label_fontsize": 23,
            "tick_fontsize": 22,
            "legend_fontsize": 22,
            "linewidth": 3.5,
        }

    run_plot_dir = plots_dir / f"seed_{seed if seed >= 0 else run_dir.name}"
    y_true = preds["y_true"]
    y_prob = preds["probs_fused"]
    class_names = getattr(eval_module, "TARGET_OBJECTS", TARGET_OBJECTS)

    eval_module.plot_multilabel_pr_curves(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        title=f"Fused MoE PR Curves ({attack})",
        path=run_plot_dir / "pr_fused_all_objects.png",
        plot_cfg=plot_cfg,
    )
    eval_module.plot_multilabel_roc_curves(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        title=f"Fused MoE ROC Curves ({attack})",
        path=run_plot_dir / "roc_fused_all_objects.png",
        plot_cfg=plot_cfg,
    )
    eval_module.plot_reliability_curve(
        y_true,
        y_prob,
        title=f"Fused MoE Reliability ({attack})",
        path=run_plot_dir / "reliability_fused.png",
        plot_cfg=plot_cfg,
    )


def aggregate_and_write(eval_module, csv_dir: Path, prefix: str, rows: List[Dict[str, Any]], key_cols: Sequence[str]) -> None:
    eval_module.write_csv(csv_dir / f"{prefix}_aggregate.csv", eval_module.summarize_table(rows, key_cols=key_cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="BDD100K MoE decision-based router attack evaluation")
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--test-script", type=str, default=DEFAULT_TEST_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--run-glob", type=str, default=DEFAULT_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack", choices=["hsj", "HSJ", "boundary"], default="hsj")
    parser.add_argument("--eps", type=float, default=0.30, help="Foolbox epsilon. Use a negative value for attacks without epsilons.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--queries", type=int, default=2500, help="Reported upper-bound query budget for compatibility with GTSRB outputs.")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_foolbox_imports()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_train_import_dependencies()
    train_module = import_module_from_path("bdd100k_moe_train_attack_module", Path(args.train_script).resolve())
    eval_module = import_module_from_path("bdd100k_moe_test_attack_module", Path(args.test_script).resolve())
    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    ensure_dir(csv_dir)
    ensure_dir(plots_dir)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    attack_name = args.attack.lower()
    eps = None if args.eps < 0 else float(args.eps)

    bundle = load_json(metadata_dir / "metadata_bundle.json")
    rows = load_json(metadata_dir / "test_fixed.json")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]

    run_dirs = list_run_dirs(args.run_dirs, args.run_glob)
    print(f"[{timestamp()}] Evaluating {len(run_dirs)} run(s) on {len(rows)} image(s).")
    print(f"[{timestamp()}] Attack={attack_name}, eps={eps}, steps={args.steps}")

    all_tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    robustness_rows: List[Dict[str, Any]] = []
    degradation: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        print(f"[{timestamp()}] Run: {run_dir}")
        model, config = load_model_for_run(run_dir, args.checkpoint_name, train_module, bundle, device)
        normalized_moe = NormalizedMoE(model).to(device).eval()
        router = RouterOnly(normalized_moe).to(device).eval()

        image_size = int(config.get("image_size", 224))
        pixel_dataset = BDD100KPixelDataset(rows, image_size=image_size)
        clean_loader = make_loader(pixel_dataset, args.batch_size, args.num_workers, device)

        clean_router_labels = collect_clean_router_labels(
            router,
            clean_loader,
            device,
            progress_label=f"{run_dir.name} clean router labels",
        )
        adv_images, success = craft_adversarial_images(
            router=router,
            loader=clean_loader,
            clean_router_labels=clean_router_labels,
            device=device,
            attack_name=attack_name,
            eps=eps,
            steps=args.steps,
            progress_label=f"{run_dir.name} {attack_name} attack",
        )
        adv_dataset = DictTensorDataset(adv_images, pixel_dataset)
        adv_loader = make_loader(adv_dataset, args.batch_size, args.num_workers, device)

        clean_preds = collect_predictions(
            normalized_moe,
            clean_loader,
            device,
            amp_dtype,
            progress_label=f"{run_dir.name} clean evaluation",
        )
        adv_preds = collect_predictions(
            normalized_moe,
            adv_loader,
            device,
            amp_dtype,
            progress_label=f"{run_dir.name} adversarial evaluation",
        )

        seed = int(config.get("seed", -1))
        clean_tables = compute_eval_tables(
            clean_preds, eval_module, weather_names, scene_names, time_names,
            args.threshold, seed, run_dir, attack_name, "clean",
        )
        adv_tables = compute_eval_tables(
            adv_preds, eval_module, weather_names, scene_names, time_names,
            args.threshold, seed, run_dir, attack_name, "adv",
        )

        for key, value in clean_tables.items():
            all_tables[f"clean_{key}"].extend(value)
        for key, value in adv_tables.items():
            all_tables[f"adv_{key}"].extend(value)

        robustness_rows.extend(routing_flip_rows(clean_preds, adv_preds, success, seed, run_dir, attack_name))
        degradation.extend(degradation_rows(clean_tables, adv_tables, seed, run_dir, attack_name))
        margin_rows.extend(margin_plot_rows_and_files(clean_preds, adv_preds, plots_dir, seed, run_dir, attack_name))
        fused_attack_comparison_plots(adv_preds, eval_module, plots_dir, seed, run_dir, attack_name)

    aggregate_and_write(eval_module, csv_dir, "adv_fused_summary", all_tables["adv_fused_summary"], ["name", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_fused_per_class", all_tables["adv_fused_per_class"], ["head", "object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_expert_per_class", all_tables["adv_expert_per_class"], ["head", "object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_expert_specialist", all_tables["adv_expert_specialist"], ["expert", "subset", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_per_class", all_tables["adv_router_per_class"], ["object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_condition", all_tables["adv_router_condition"], ["task", "subset", "attack", "split"])

    eval_module.write_csv(csv_dir / "robustness_summary_aggregate.csv", eval_module.summarize_table(robustness_rows, key_cols=["attack", "subset"]))
    eval_module.write_csv(csv_dir / "metric_degradation_aggregate.csv", eval_module.summarize_table(degradation, key_cols=["attack", "component", "metric"]))

    save_json(
        {
            "attack": attack_name,
            "eps": eps,
            "steps": args.steps,
            "queries_reported_upper_bound": args.queries,
            "num_samples": len(rows),
            "run_dirs": [str(p) for p in run_dirs],
            "checkpoint_name": args.checkpoint_name,
            "router_label_definition": "clean_router_top1",
        },
        output_dir / "decision_attack_manifest.json",
    )

    print(f"[{timestamp()}] Decision-based attack evaluation complete.")
    print(f"[{timestamp()}] CSV files saved to: {csv_dir}")
    print(f"[{timestamp()}] Margin/drop plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
