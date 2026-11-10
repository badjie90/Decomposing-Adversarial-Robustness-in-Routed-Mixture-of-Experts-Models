#!/usr/bin/env python3
"""Train the custom sparse token-routed control model on the same multilabel task.

This is a deliberately small, transparent top-k token MoE control, not an
official V-MoE/Switch release.  It provides a structurally different routing
mechanism for the requested cross-architecture experiment.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PixelDataset, disjoint_validation_rows, load_metadata, split_audit
from .io import save_json
from .metrics import prediction_metrics
from .sparse_model import SparseVisionMoE


from .paths import metadata_dir

DEFAULT_METADATA = {dataset: metadata_dir(dataset) for dataset in ("bdd100k", "nuimages")}


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {"image": torch.stack([x["image"] for x in batch]), "labels": torch.stack([x["labels"] for x in batch])}


def evaluate(model, loader, device, max_batches=0):
    model.eval(); y, p, losses = [], [], []
    with torch.no_grad():
        for count, batch in enumerate(loader, 1):
            image, labels = batch["image"].to(device), batch["labels"].to(device)
            out = model.forward_components(image)
            losses.append(float(F.binary_cross_entropy_with_logits(out["fused_logits"], labels).cpu()))
            y.append(labels.cpu().numpy().astype(np.int8)); p.append(out["fused_logits"].sigmoid().cpu().numpy())
            if max_batches and count >= max_batches: break
    result = prediction_metrics(np.concatenate(y), np.concatenate(p)); result["bce"] = float(np.mean(losses)); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bdd100k", "nuimages"), required=True)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--balance-weight", type=float, default=.01)
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--patch-size", type=int, default=16); parser.add_argument("--embed-dim", type=int, default=128); parser.add_argument("--num-experts", type=int, default=4); parser.add_argument("--top-k", type=int, default=2); parser.add_argument("--depth", type=int, default=2); parser.add_argument("--num-heads", type=int, default=4); parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--early-stopping-patience", type=int, default=10); parser.add_argument("--max-train-batches", type=int, default=0); parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args(); args.metadata_dir = (args.metadata_dir or DEFAULT_METADATA[args.dataset]).resolve(); args.output_dir = args.output_dir.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): parser.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir(); seed_everything(args.seed); device = torch.device(args.device)
    train, bundle = load_metadata(args.dataset, str(args.metadata_dir), "train")
    if args.dataset == "bdd100k": val, val_audit = disjoint_validation_rows(args.dataset, str(args.metadata_dir))
    else: val, _ = load_metadata(args.dataset, str(args.metadata_dir), "val"); val_audit = split_audit(args.dataset, str(args.metadata_dir))
    train_ds = PixelDataset(train, args.image_size, args.dataset, bundle); val_ds = PixelDataset(val, args.image_size, args.dataset, bundle)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate)
    config = {key: value for key, value in vars(args).items() if key not in ("metadata_dir", "output_dir")}
    model = SparseVisionMoE(num_classes=3, image_size=args.image_size, patch_size=args.patch_size, embed_dim=args.embed_dim, num_experts=args.num_experts, top_k=args.top_k, depth=args.depth, num_heads=args.num_heads, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda"); best = -np.inf; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for count, batch in enumerate(train_loader, 1):
            image, labels = batch["image"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model.forward_components(image); bce = F.binary_cross_entropy_with_logits(output["fused_logits"], labels); balance = model.balance_loss(output); loss = bce + args.balance_weight * balance
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            if args.max_train_batches and count >= args.max_train_batches: break
        validation = evaluate(model, val_loader, device, args.max_val_batches); schedule.step()
        limited = bool(args.max_train_batches or args.max_val_batches)
        if np.isfinite(validation["macro_auroc"]):
            score, selection_metric = validation["macro_auroc"], "macro_auroc"
        elif limited and np.isfinite(validation["bce"]):
            score, selection_metric = -validation["bce"], "negative_bce_smoke_only"
        else:
            score, selection_metric = float("nan"), "macro_auroc"
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation, "selection_score": score, "selection_metric": selection_metric, "lr": optimizer.param_groups[0]["lr"]}; history.append(row); save_json(args.output_dir / "history.json", history)
        print({"epoch": epoch, "train_loss": row["train_loss"], "validation_macro_auroc": validation["macro_auroc"], "selection_metric": selection_metric}, flush=True)
        payload = {"model": model.state_dict(), "model_config": model.config_dict, "dataset": args.dataset, "config": config, "validation_audit": val_audit, "epoch": epoch}
        torch.save(payload, args.output_dir / "checkpoints" / "last.pt")
        if np.isfinite(score) and score > best:
            best = score; stale = 0; torch.save(payload, args.output_dir / "checkpoints" / "best.pt")
        else: stale += 1
        if stale >= args.early_stopping_patience: break
    if not (args.output_dir / "checkpoints" / "best.pt").exists(): raise RuntimeError("No sparse checkpoint selected")
    save_json(args.output_dir / "COMPLETED.json", {"best_validation_macro_auroc": best, "epochs": len(history), "custom_architecture": True, "not_official_vmoe_or_switch": True})


if __name__ == "__main__": main()
