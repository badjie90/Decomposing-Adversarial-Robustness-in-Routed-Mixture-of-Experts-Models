"""Custom sparse patch-token vision MoE for architecture controls.

This is NOT an official V-MoE or Switch Transformer implementation/checkpoint.
A shared patch embedding and attention encoder feed per-token top-k FFN experts.
The global classifier receives pooled routed token features. No context labels
supervise its experts. Capture mode evaluates every expert to permit exact
counterfactual swaps; predict_sparse dispatches only selected token/expert pairs.

Image-level alpha is mean token_alpha, and router_logits is log(alpha), not the
raw token-router logit. This preserves argmax consistency for image-level attacks
and diagnostics. Raw token logits and sparse routing maps are also returned and
must be saved. Attribution swaps token maps, never their image-level averages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn

from .adapters import IMAGENET_MEAN, IMAGENET_STD, file_sha256


class TokenExpert(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, tokens):
        return tokens + self.net(tokens)


class SparseVisionMoE(nn.Module):
    architecture = "custom_sparse_token_moe"
    routing_unit = "patch_token"

    def __init__(self, num_classes: int = 3, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 128,
                 num_experts: int = 4, top_k: int = 2, depth: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts.")
        if image_size % patch_size or embed_dim % num_heads:
            raise ValueError("image_size and embed_dim must divide patch_size and num_heads.")
        self.config_dict = {"num_classes": num_classes, "image_size": image_size,
                            "patch_size": patch_size, "embed_dim": embed_dim,
                            "num_experts": num_experts, "top_k": top_k,
                            "depth": depth, "num_heads": num_heads, "dropout": dropout}
        self.num_experts, self.top_k = num_experts, top_k
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=4 * embed_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth, enable_nested_tensor=False)
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([TokenExpert(embed_dim, dropout) for _ in range(num_experts)])
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def encode(self, pixels):
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} pixel inputs.")
        tokens = self.patch_embed((pixels - self.mean) / self.std).flatten(2).transpose(1, 2)
        return self.encoder(tokens + self.position)

    def route(self, features):
        logits = self.router(features)
        probabilities = logits.softmax(-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        if self.top_k == 1:
            # Straight-through gate: hard one-hot forward, soft gradient backward.
            # Top-2 is the default; top-1's ST estimator is recorded in config.
            hard = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
            token_alpha = hard + probabilities - probabilities.detach()
        else:
            values = values / values.sum(-1, keepdim=True)
            token_alpha = torch.zeros_like(probabilities).scatter(-1, indices, values)
        alpha = token_alpha.mean(dim=1)
        pooled_logits = alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()
        return logits, token_alpha, alpha, pooled_logits

    def forward_components(self, pixels) -> Dict[str, torch.Tensor]:
        features = self.encode(pixels)
        token_logits, token_alpha, alpha, pooled_logits = self.route(features)
        # [B,T,K,D]. Dense capture is required even for inactive token experts,
        # since a routing intervention can activate a previously inactive one.
        expert_features = torch.stack([expert(features) for expert in self.experts], dim=2)
        expert_logits = self.head(expert_features.mean(dim=1))
        return {"features": features, "router_logits": pooled_logits, "alpha": alpha,
                "expert_features": expert_features, "expert_logits": expert_logits,
                "fused_logits": self.fuse(expert_features, token_alpha),
                "token_router_logits": token_logits, "token_alpha": token_alpha}

    def forward(self, pixels):
        return self.forward_components(pixels)

    def fuse(self, expert_features, alpha):
        if expert_features.ndim != 4 or alpha.shape != expert_features.shape[:3]:
            raise ValueError("Token fusion requires expert_features [B,T,K,D] and FULL alpha [B,T,K].")
        routed = (expert_features * alpha.unsqueeze(-1)).sum(dim=2)
        return self.head(routed.mean(dim=1))

    def attack_outputs(self, pixels, target_kind):
        if target_kind == "router":
            return self.route(self.encode(pixels))[-1]
        if target_kind == "fused":
            return self.predict_sparse(pixels)
        raise ValueError(f"Unknown attack target {target_kind!r}")

    def predict_sparse(self, pixels):
        """Actual sparse dispatch; numerically matches capture mode in eval()."""
        features = self.encode(pixels)
        _, weights, _, _ = self.route(features)
        flat = features.reshape(-1, features.shape[-1])
        flat_weights = weights.reshape(-1, self.num_experts)
        routed = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            selected = torch.nonzero(flat_weights[:, index] > 0, as_tuple=False).flatten()
            if selected.numel():
                contribution = expert(flat[selected]) * flat_weights[selected, index:index + 1]
                routed = routed.index_add(0, selected, contribution)
        return self.head(routed.reshape_as(features).mean(dim=1))

    def balance_loss(self, components):
        """Switch-style load balancing on token probabilities/selected fractions."""
        probabilities = components["token_router_logits"].softmax(-1)
        selected = (components["token_alpha"].detach() > 0).to(probabilities.dtype) / self.top_k
        return self.num_experts * (probabilities.mean((0, 1)) * selected.mean((0, 1))).sum()


def load_sparse_model(checkpoint_path: str, device: str = "cpu") -> Tuple[SparseVisionMoE, Dict[str, Any]]:
    from .data import class_names_for

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" not in payload:
        raise ValueError("Sparse checkpoint requires model_config and model keys.")
    model = SparseVisionMoE(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    dataset = payload.get("dataset", payload.get("config", {}).get("dataset"))
    classes = class_names_for(dataset) if dataset else [f"class_{i}" for i in range(model.config_dict["num_classes"])]
    info = {"architecture": model.architecture, "model_config": model.config_dict,
            "config": payload.get("config", {}), "image_size": model.image_size,
            "checkpoint_path": str(path), "checkpoint_sha256": file_sha256(path),
            "class_names": classes, "expert_names": [f"token_expert_{i}" for i in range(model.num_experts)],
            "routing_unit": model.routing_unit,
            "image_routing_summary": "mean sparse token weights; log of this mean used as image router scores",
            "top1_gradient_estimator": "straight_through_softmax" if model.top_k == 1 else "not_applicable",
            "implementation": "Custom architecture; not an official V-MoE or Switch checkpoint"}
    return model, info
