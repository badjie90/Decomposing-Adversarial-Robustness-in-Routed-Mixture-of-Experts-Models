#!/usr/bin/env python3
"""Retrain BDD100K CSA-GMoE with validation images disjoint from the fixed test set.

The historical BDD100K ``val.json`` contains the full 10,000-image test cohort.
This trainer removes every train/test image from validation before selecting a
checkpoint.  It reuses the unmodified original CSA-GMoE architecture and loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .adapters import file_sha256, original_training_module
from .data import disjoint_validation_rows, load_metadata, split_audit
from .io import save_json
from .metrics import prediction_metrics


from .paths import metadata_dir

DEFAULT_METADATA = metadata_dir("bdd100k")


def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _loader(module, rows, transform, args, *, shuffle: bool, weighted: bool, generator: torch.Generator):
    dataset = module.BDD100KPresenceDataset(rows, transform)
    sampler = None
    if weighted:
        sampler = WeightedRandomSampler(module.compute_sample_weights(rows), len(rows), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
                      worker_init_fn=module.seed_worker, generator=generator)


def _run_epoch(module, model, loader, optimizer, scaler, stage, pos_weight, device, amp_dtype, grad_clip_norm, max_batches=0):
    model.train(); logs: dict[str, list[float]] = {}
    for number, batch in enumerate(loader, 1):
        batch = module.move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            output = model(batch["image"])
            loss, row = module.compute_losses(stage, output, batch, pos_weight, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer); scaler.update()
        for key, value in row.items(): logs.setdefault(key, []).append(float(value))
        if max_batches and number >= max_batches: break
    return {key: float(np.mean(value)) for key, value in logs.items()}


@torch.no_grad()
def _evaluate(module, model, loader, stage, pos_weight, device, amp_dtype, max_batches=0):
    model.eval(); logs: dict[str, list[float]] = {}; labels = []; probs = []
    for number, batch in enumerate(loader, 1):
        batch = module.move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            output = model(batch["image"])
            _, row = module.compute_losses(stage, output, batch, pos_weight, device)
        for key, value in row.items(): logs.setdefault(key, []).append(float(value))
        labels.append(batch["objects"].cpu().numpy().astype(np.int8))
        probs.append(output["obj_fused_logits"].sigmoid().float().cpu().numpy())
        if max_batches and number >= max_batches: break
    metrics = {"loss_" + key: float(np.mean(value)) for key, value in logs.items()}
    metrics.update(prediction_metrics(np.concatenate(labels), np.concatenate(probs)))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-name", default="convnextv2_tiny.fcmae_ft_in1k")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-test only: cap batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Smoke-test only: cap validation batches per epoch.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device); args.output_dir = args.output_dir.resolve(); args.metadata_dir = args.metadata_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}. Use a new directory to avoid overwriting a run.")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir()
    set_all_seeds(args.seed)
    module = original_training_module("bdd100k")
    train_rows, bundle = load_metadata("bdd100k", str(args.metadata_dir), "train")
    val_rows, validation_audit = disjoint_validation_rows("bdd100k", str(args.metadata_dir))
    audit = split_audit("bdd100k", str(args.metadata_dir))
    if not audit["overlap"]["val__test_fixed"]["count"]:
        raise RuntimeError("Expected to audit the known historical overlap, but none was found; inspect metadata before training.")
    generator = torch.Generator().manual_seed(args.seed)
    train_tf, eval_tf = module.build_transforms(args.image_size)
    train_loader = _loader(module, train_rows, train_tf, args, shuffle=True, weighted=args.weighted_sampling, generator=generator)
    val_loader = _loader(module, val_rows, eval_tf, args, shuffle=False, weighted=False, generator=generator)
    model = module.build_model("stage3_moe", args.backbone_name, len(bundle["weather_to_id"]), len(bundle["scene_to_id"]), len(bundle["time_to_id"]), pretrained=not args.no_pretrained).to(device)
    optimizer = module.build_optimizer(model, "stage3_moe", args.backbone_lr, args.expert_lr, args.head_lr, args.weight_decay)
    scheduler = module.build_scheduler(optimizer, args.epochs, args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    pos_weight = module.compute_object_pos_weight(train_rows).to(device)
    # Keep checkpoint metadata restricted to JSON-like values. In particular,
    # serializing argparse ``Path`` objects would make PyTorch 2.6+ safe
    # weights-only loading reject an otherwise ordinary model checkpoint.
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"stage": "stage3_moe", "device": str(device), "weather_to_id": bundle["weather_to_id"],
        "scene_to_id": bundle["scene_to_id"], "time_to_id": bundle["time_to_id"], "original_training_source": str(Path(module.__file__).resolve()),
        "original_training_source_sha256": file_sha256(Path(module.__file__)), "validation_audit": validation_audit,
        "historical_split_audit": audit, "test_set_used_for_training_or_selection": False})
    save_json(args.output_dir / "config.json", config); save_json(args.output_dir / "split_audit.json", audit)
    best = -math.inf; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(module, model, train_loader, optimizer, scaler, "stage3_moe", pos_weight, device, amp_dtype, args.grad_clip_norm, args.max_train_batches)
        val = _evaluate(module, model, val_loader, "stage3_moe", pos_weight, device, amp_dtype, args.max_val_batches)
        # A deliberately tiny smoke validation subset can lack a positive or a
        # negative class, making strict macro AUROC undefined.  Full runs still
        # select by macro AUROC; smoke runs use finite BCE only to exercise the
        # checkpoint path without pretending to validate a model.
        smoke_limited = bool(args.max_train_batches or args.max_val_batches)
        if np.isfinite(val["macro_auroc"]):
            score, selection_metric = val["macro_auroc"], "macro_auroc"
        elif smoke_limited and np.isfinite(val["bce"]):
            score, selection_metric = -val["bce"], "negative_bce_smoke_only"
        else:
            score, selection_metric = float("nan"), "macro_auroc"
        scheduler.step()
        row = {"epoch": epoch, "train": train, "validation": val, "selection_metric": selection_metric, "selection_score": score, "lr": [x["lr"] for x in optimizer.param_groups]}; history.append(row)
        save_json(args.output_dir / "history.json", history)
        print(json.dumps({"epoch": epoch, "train_loss": train.get("loss_total"), "validation_macro_auroc": val.get("macro_auroc"), "selection_metric": selection_metric, "selection_score": score, "validation_macro_f1": val.get("macro_f1")}), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "best_validation_macro_auroc": best, "config": config}
        torch.save(payload, args.output_dir / "checkpoints" / "last.pt")
        if np.isfinite(score) and score > best + args.early_stopping_min_delta:
            best = score; stale = 0
            payload["best_validation_macro_auroc"] = best
            torch.save(payload, args.output_dir / "checkpoints" / "best.pt")
        else:
            stale += 1
        if stale >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch}; best disjoint-validation macro AUROC={best:.6f}")
            break
    if not (args.output_dir / "checkpoints" / "best.pt").exists():
        raise RuntimeError("No valid checkpoint was selected; inspect validation class coverage/configuration.")
    save_json(args.output_dir / "COMPLETED.json", {"best_selection_score": best, "selection_metric": history[-1]["selection_metric"], "epochs_completed": len(history), "smoke_run": bool(args.max_train_batches or args.max_val_batches)})


if __name__ == "__main__":
    main()
