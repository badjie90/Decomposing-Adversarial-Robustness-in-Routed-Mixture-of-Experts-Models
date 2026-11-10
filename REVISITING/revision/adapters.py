"""Exact CSA-GMoE inference with cached routing interventions.

Inputs are resized/cropped RGB pixels in [0, 1]. ImageNet normalization happens
inside this adapter, so perturbation budgets are always expressed in pixel units.
The original learned fused head is applied to a weighted sum of expert FEATURES.
The three per-expert object heads are diagnostics, not the fusion operator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PosixPath
from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_training_module(dataset: str):
    """Import the existing source without changing or copying its architecture."""
    dataset = dataset.lower()
    paths = {
        "bdd100k": BASELINE_ROOT / "bdd100k_moe_train.py",
        "nuimages": BASELINE_ROOT / "nuimages_moe_train.py",
    }
    if dataset not in paths:
        raise ValueError(f"Unknown dataset: {dataset}")
    path = paths[dataset]
    name = f"_revision_original_{dataset}_train"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import original architecture: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return sys.modules[name]


class CSAAdapter(nn.Module):
    """Expose the original model's components and an independent fusion method."""

    architecture = "csa_gmoe"
    routing_unit = "image"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels - self.mean) / self.std

    def forward_components(self, pixels: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.model.backbone(self.normalize(pixels))
        experts = torch.stack([
            self.model.weather_expert(features),
            self.model.scene_expert(features),
            self.model.time_expert(features),
        ], dim=1)
        router_logits = self.model.router(features)
        alpha = router_logits.softmax(dim=-1)
        expert_logits = torch.stack([
            self.model.obj_head_weather(experts[:, 0]),
            self.model.obj_head_scene(experts[:, 1]),
            self.model.obj_head_time(experts[:, 2]),
        ], dim=1)
        return {
            "features": features,
            "router_logits": router_logits,
            "alpha": alpha,
            "expert_features": experts,
            "expert_logits": expert_logits,
            "fused_logits": self.fuse(experts, alpha),
        }

    def forward(self, pixels: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_components(pixels)

    def attack_outputs(self, pixels: torch.Tensor, target_kind: str) -> torch.Tensor:
        """Router queries skip experts; fused queries preserve original fusion."""
        if target_kind == "router":
            return self.model.router(self.model.backbone(self.normalize(pixels)))
        if target_kind == "fused":
            return self.forward_components(pixels)["fused_logits"]
        raise ValueError(f"Unknown attack target {target_kind!r}")

    def fuse(self, expert_features: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if expert_features.ndim != 3 or alpha.shape != expert_features.shape[:2]:
            raise ValueError("Expected expert_features [B,3,D] and alpha [B,3].")
        if expert_features.shape[1] != 3:
            raise ValueError("CSA-GMoE has exactly three expert slots.")
        # Keep original operation ordering, including floating-point additions.
        fused = (alpha[:, 0:1] * expert_features[:, 0]
                 + alpha[:, 1:2] * expert_features[:, 1]
                 + alpha[:, 2:3] * expert_features[:, 2])
        return self.model.fused_obj_head(fused)


def controlled_conditions(
    adapter: nn.Module,
    clean: Mapping[str, torch.Tensor],
    adversarial: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """CC/CA/AC/AA logits from cached tensors; never calls encoder or experts.

    Token models preserve the entire [B,T,K] routing map in every intervention.
    Pooled routing weights are diagnostic only and must never replace this map.
    Call in eval mode so dropout cannot contaminate controlled comparisons.
    """
    if adapter.training:
        raise ValueError("Controlled interventions require adapter.eval().")
    routing_key = "token_alpha" if "token_alpha" in clean else "alpha"
    if (routing_key in adversarial) is False:
        raise ValueError("Clean and adversarial routing representations differ.")
    if clean["expert_features"].shape != adversarial["expert_features"].shape:
        raise ValueError("Clean and adversarial expert tensors must be paired.")
    return {
        "CC": adapter.fuse(clean["expert_features"], clean[routing_key]),
        "CA": adapter.fuse(clean["expert_features"], adversarial[routing_key]),
        "AC": adapter.fuse(adversarial["expert_features"], clean[routing_key]),
        "AA": adapter.fuse(adversarial["expert_features"], adversarial[routing_key]),
    }


def load_csa_model(
    dataset: str,
    run_dir: str,
    metadata_dir: str,
    device: str = "cpu",
    checkpoint_name: str = "best.pt",
) -> Tuple[CSAAdapter, Dict[str, Any]]:
    from .data import class_names_for, expert_names_for, load_metadata

    run_dir = Path(run_dir).expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    _, bundle = load_metadata(dataset, metadata_dir, split=None)
    if config.get("stage") == "stage0_baseline":
        raise ValueError("The baseline checkpoint has no router.")
    for key in ("weather_to_id", "scene_to_id", "time_to_id"):
        if key in config and config[key] != bundle[key]:
            raise ValueError(f"Checkpoint/metadata class mapping mismatch: {key}")
    module = original_training_module(dataset)
    model = module.build_model(
        stage=config.get("stage", "stage3_moe"),
        backbone_name=config["backbone_name"],
        num_weather=len(bundle["weather_to_id"]),
        num_scene=len(bundle["scene_to_id"]),
        num_time=len(bundle["time_to_id"]),
        pretrained=False,
    )
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    # New checkpoints record their training configuration alongside weights.
    # ``argparse`` represents path arguments as ``pathlib.PosixPath`` objects,
    # which PyTorch 2.6+ refuses by default in weights-only mode.  This is a
    # standard-library value, not executable code; allow only that narrow type
    # while retaining safe weights-only deserialization for model state.
    with torch.serialization.safe_globals([PosixPath]):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    adapter = CSAAdapter(model).to(device).eval()
    info = {
        "architecture": adapter.architecture,
        "config": config,
        "class_names": class_names_for(dataset),
        "expert_names": expert_names_for(dataset),
        "image_size": int(config.get("image_size", 224)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "original_model_source": str(Path(module.__file__).resolve()),
        "original_model_source_sha256": file_sha256(Path(module.__file__)),
        "metadata": bundle,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "perturbation_space": "RGB pixel values [0,1] after resize/center crop, before normalization",
    }
    return adapter, info
