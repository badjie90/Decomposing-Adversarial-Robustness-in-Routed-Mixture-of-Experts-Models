#!/usr/bin/env python3
# path: nuimages_moe_train.py
"""nuImages CSA-GMoE architecture, training and metadata preparation.

The three internal condition slots represent location, illumination and motion.
The object outputs represent car, pedestrian and traffic cone. The third object
label retains the legacy traffic_sign_present key for checkpoint compatibility.
The official validation set supplies the fixed test set. Model selection uses a
seeded ten-percent holdout from the official training set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required. Install with: pip install timm") from exc

try:
    from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MulticlassAccuracy
except ImportError as exc:
    raise ImportError("torchmetrics is required. Install with: pip install torchmetrics") from exc

try:
    from torchvision import transforms
except ImportError as exc:
    raise ImportError("torchvision is required. Install with: pip install torchvision") from exc

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


ImageFile.LOAD_TRUNCATED_IMAGES = True

# These names preserve the original 3-logit structure.
TARGET_OBJECTS = ["car", "pedestrian", "traffic_cone"]
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_LOCATION_TO_TIMEZONE = {
    "boston-seaport": "America/New_York",
    "singapore-onenorth": "Asia/Singapore",
    "singapore-hollandvillage": "Asia/Singapore",
    "singapore-queenstown": "Asia/Singapore",
}

DEFAULT_ILLUMINATION_TO_ID = {
    "day": 0,
    "twilight": 1,
    "night": 2,
}

DEFAULT_MOTION_TO_ID = {
    "stopped": 0,
    "urban": 1,
    "fast": 2,
}


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(REPOSITORY_ROOT / "data/nuimages")
DEFAULT_TRAIN_VERSION = "nuimages-metadata/v1.0-train"
DEFAULT_VAL_VERSION = "nuimages-metadata/v1.0-val"
DEFAULT_OUTPUT_DIR = str(REPOSITORY_ROOT / "data/metadata/nuimages")
DEFAULT_EXPERIMENT_DIR = str(REPOSITORY_ROOT / "results/models/nuimages/csa_seed42")



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class MetadataRow:
    image_path: str
    split: str
    weather_id: int
    scene_id: int
    time_id: int
    weather_mask: int
    scene_mask: int
    time_mask: int
    car_present: int
    pedestrian_present: int
    traffic_sign_present: int


@dataclass
class MetadataBundle:
    weather_to_id: Dict[str, int]
    scene_to_id: Dict[str, int]
    time_to_id: Dict[str, int]
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


def load_table(version_root: Path, table_name: str) -> List[Dict[str, Any]]:
    path = version_root / f"{table_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing nuImages table: {path}")
    return load_json(path)


def index_by_token(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["token"]: row for row in rows}


def group_object_annotations_by_sample_data(object_ann_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in object_ann_rows:
        grouped.setdefault(row["sample_data_token"], []).append(row)
    return grouped


def infer_timestamp_seconds(raw_ts: int) -> float:
    # nuImages timestamps are typically in microseconds.
    if raw_ts > 10**14:
        return raw_ts / 1e6
    if raw_ts > 10**11:
        return raw_ts / 1e3
    return float(raw_ts)


def derive_illumination_label(timestamp_value: int, location_name: str) -> str:
    seconds = infer_timestamp_seconds(int(timestamp_value))
    dt_utc = datetime.fromtimestamp(seconds, tz=timezone.utc)

    tz_name = DEFAULT_LOCATION_TO_TIMEZONE.get(location_name)
    if tz_name and ZoneInfo is not None:
        try:
            dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        except Exception:
            dt_local = dt_utc
    else:
        dt_local = dt_utc

    hour = dt_local.hour
    if 7 <= hour < 18:
        return "day"
    if 5 <= hour < 7 or 18 <= hour < 20:
        return "twilight"
    return "night"


def derive_motion_label(speed_mps: float) -> str:
    if speed_mps < 0.5:
        return "stopped"
    if speed_mps < 8.0:
        return "urban"
    return "fast"


def is_car_category(category_name: str) -> bool:
    return category_name == "vehicle.car"


def is_pedestrian_category(category_name: str) -> bool:
    return category_name.startswith("human.pedestrian.")


def is_traffic_cone_category(category_name: str) -> bool:
    return category_name == "movable_object.trafficcone"


def derive_object_presence(
    anns_for_sd: Sequence[Dict[str, Any]],
    category_by_token: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, int]:
    car_present = 0
    pedestrian_present = 0
    traffic_cone_present = 0

    for ann in anns_for_sd:
        category = category_by_token.get(ann["category_token"])
        if category is None:
            continue
        category_name = category["name"]

        if is_car_category(category_name):
            car_present = 1
        if is_pedestrian_category(category_name):
            pedestrian_present = 1
        if is_traffic_cone_category(category_name):
            traffic_cone_present = 1

    return car_present, pedestrian_present, traffic_cone_present


def compute_split_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    stats: Dict[str, Any] = {"num_samples": len(rows), "objects": {}, "masks": {}}

    for obj_name, key in [
        ("car", "car_present"),
        ("pedestrian", "pedestrian_present"),
        ("traffic_cone", "traffic_sign_present"),
    ]:
        positives = sum(int(row[key]) for row in rows)
        stats["objects"][obj_name] = {
            "positive": positives,
            "negative": len(rows) - positives,
            "positive_ratio": positives / max(1, len(rows)),
        }

    for task_name in ("weather", "scene", "time"):
        mask_key = f"{task_name}_mask"
        valid = sum(int(row[mask_key]) for row in rows)
        stats["masks"][task_name] = {
            "valid": valid,
            "invalid": len(rows) - valid,
            "valid_ratio": valid / max(1, len(rows)),
        }

    return stats


def split_train_val(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def build_rows_for_version(
    data_root: Path,
    version_root: Path,
    split_name: str,
    location_to_id: Dict[str, int],
    illumination_to_id: Dict[str, int],
    motion_to_id: Dict[str, int],
) -> List[Dict[str, Any]]:
    sample_rows = load_table(version_root, "sample")
    sample_data_rows = load_table(version_root, "sample_data")
    object_ann_rows = load_table(version_root, "object_ann")
    category_rows = load_table(version_root, "category")
    ego_pose_rows = load_table(version_root, "ego_pose")
    log_rows = load_table(version_root, "log")

    sample_data_by_token = index_by_token(sample_data_rows)
    category_by_token = index_by_token(category_rows)
    ego_pose_by_token = index_by_token(ego_pose_rows)
    log_by_token = index_by_token(log_rows)
    object_ann_by_sd = group_object_annotations_by_sample_data(object_ann_rows)

    rows: List[Dict[str, Any]] = []

    for sample in sample_rows:
        key_camera_token = sample["key_camera_token"]
        sample_data = sample_data_by_token.get(key_camera_token)
        if sample_data is None or not sample_data.get("is_key_frame", False):
            continue

        image_path = data_root / sample_data["filename"]
        if not image_path.exists():
            continue

        log_record = log_by_token[sample["log_token"]]
        ego_pose = ego_pose_by_token[sample_data["ego_pose_token"]]

        location_label = str(log_record["location"]).strip().lower()
        illumination_label = derive_illumination_label(sample_data["timestamp"], location_label)
        motion_label = derive_motion_label(float(ego_pose.get("speed", 0.0)))

        car_present, pedestrian_present, traffic_cone_present = derive_object_presence(
            anns_for_sd=object_ann_by_sd.get(key_camera_token, []),
            category_by_token=category_by_token,
        )

        rows.append(
            asdict(
                MetadataRow(
                    image_path=str(image_path.resolve()),
                    split=split_name,
                    weather_id=location_to_id[location_label],
                    scene_id=illumination_to_id[illumination_label],
                    time_id=motion_to_id[motion_label],
                    weather_mask=1,
                    scene_mask=1,
                    time_mask=1,
                    car_present=car_present,
                    pedestrian_present=pedestrian_present,
                    traffic_sign_present=traffic_cone_present,
                )
            )
        )

    return rows


def build_metadata(
    data_root: Path,
    train_version_root: Path,
    val_version_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> MetadataBundle:
    train_logs = load_table(train_version_root, "log")
    val_logs = load_table(val_version_root, "log")

    all_locations = sorted(
        {
            str(log["location"]).strip().lower()
            for log in list(train_logs) + list(val_logs)
        }
    )
    location_to_id = {name: idx for idx, name in enumerate(all_locations)}

    train_rows_full = build_rows_for_version(
        data_root=data_root,
        version_root=train_version_root,
        split_name="train",
        location_to_id=location_to_id,
        illumination_to_id=DEFAULT_ILLUMINATION_TO_ID,
        motion_to_id=DEFAULT_MOTION_TO_ID,
    )
    test_rows = build_rows_for_version(
        data_root=data_root,
        version_root=val_version_root,
        split_name="val",
        location_to_id=location_to_id,
        illumination_to_id=DEFAULT_ILLUMINATION_TO_ID,
        motion_to_id=DEFAULT_MOTION_TO_ID,
    )

    train_rows, extra_val_rows = split_train_val(train_rows_full, val_ratio=val_ratio, seed=seed)
    val_rows = extra_val_rows

    stats = {
        "train": compute_split_stats(train_rows),
        "val": compute_split_stats(val_rows),
        "test": compute_split_stats(test_rows),
        "mapping": {
            "weather_to_id": location_to_id,
            "scene_to_id": DEFAULT_ILLUMINATION_TO_ID,
            "time_to_id": DEFAULT_MOTION_TO_ID,
            "condition_semantics": {
                "weather": "location",
                "scene": "illumination",
                "time": "motion",
            },
            "object_semantics": {
                "slot_0": "car",
                "slot_1": "pedestrian",
                "slot_2": "traffic_cone",
            },
        },
    }

    bundle = MetadataBundle(
        weather_to_id=location_to_id,
        scene_to_id=DEFAULT_ILLUMINATION_TO_ID,
        time_to_id=DEFAULT_MOTION_TO_ID,
        train=train_rows,
        val=val_rows,
        test=test_rows,
        stats=stats,
    )

    ensure_dir(output_dir)
    save_json(asdict(bundle), output_dir / "metadata_bundle.json")
    save_json(train_rows, output_dir / "train.json")
    save_json(val_rows, output_dir / "val.json")
    save_json(test_rows, output_dir / "test_fixed.json")
    save_json(stats, output_dir / "stats.json")

    print(f"[{timestamp()}] Metadata saved to: {output_dir}")
    print(f"[{timestamp()}] Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
    return bundle


class NuImagesPresenceDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], transform: transforms.Compose) -> None:
        self.rows = list(rows)
        self.transform = transform

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
                [
                    row["car_present"],
                    row["pedestrian_present"],
                    row["traffic_sign_present"],
                ],
                dtype=torch.float32,
            ),
            "image_path": row["image_path"],
        }


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return train_transform, eval_transform


def compute_object_pos_weight(rows: Sequence[Dict[str, Any]]) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.float64)
    for row in rows:
        counts[0] += row["car_present"]
        counts[1] += row["pedestrian_present"]
        counts[2] += row["traffic_sign_present"]
    total = len(rows)
    neg = total - counts
    pos_weight = np.divide(neg, np.maximum(counts, 1.0))
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_sample_weights(rows: Sequence[Dict[str, Any]]) -> torch.Tensor:
    weather_counts: Dict[int, int] = {}
    scene_counts: Dict[int, int] = {}
    time_counts: Dict[int, int] = {}

    for row in rows:
        weather_counts[row["weather_id"]] = weather_counts.get(row["weather_id"], 0) + 1
        scene_counts[row["scene_id"]] = scene_counts.get(row["scene_id"], 0) + 1
        time_counts[row["time_id"]] = time_counts.get(row["time_id"], 0) + 1

    weights: List[float] = []
    for row in rows:
        score = 1.0
        score += 1.0 / max(1, weather_counts[row["weather_id"]])
        score += 1.0 / max(1, scene_counts[row["scene_id"]])
        score += 1.0 / max(1, time_counts[row["time_id"]])
        if row["pedestrian_present"]:
            score += 0.5
        if row["traffic_sign_present"]:
            score += 0.5
        weights.append(score)

    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np /= weights_np.mean()
    return torch.tensor(weights_np, dtype=torch.double)


class MLPExpert(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_experts: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        if hasattr(self.backbone, "num_features"):
            self.out_dim = int(self.backbone.num_features)
        elif hasattr(self.backbone, "feature_info"):
            self.out_dim = int(self.backbone.feature_info.channels()[-1])
        else:
            raise RuntimeError(f"Could not infer feature dimension for backbone: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PresenceBaseline(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = SharedBackbone(backbone_name, pretrained=pretrained)
        self.head = MLPHead(self.backbone.out_dim, out_dim=3)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(image)
        return {"obj_fused_logits": self.head(h)}


class ConditionMoEModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_weather: int,
        num_scene: int,
        num_time: int,
        pretrained: bool = True,
        expert_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = SharedBackbone(backbone_name, pretrained=pretrained)
        dim = self.backbone.out_dim

        self.weather_expert = MLPExpert(dim=dim, dropout=expert_dropout)
        self.scene_expert = MLPExpert(dim=dim, dropout=expert_dropout)
        self.time_expert = MLPExpert(dim=dim, dropout=expert_dropout)

        self.weather_head = MLPHead(dim, num_weather, dropout=head_dropout)
        self.scene_head = MLPHead(dim, num_scene, dropout=head_dropout)
        self.time_head = MLPHead(dim, num_time, dropout=head_dropout)

        self.obj_head_weather = MLPHead(dim, 3, dropout=head_dropout)
        self.obj_head_scene = MLPHead(dim, 3, dropout=head_dropout)
        self.obj_head_time = MLPHead(dim, 3, dropout=head_dropout)

        self.router = Router(dim)
        self.fused_obj_head = MLPHead(dim, 3, dropout=head_dropout)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(image)

        e_w = self.weather_expert(h)
        e_s = self.scene_expert(h)
        e_t = self.time_expert(h)

        router_logits = self.router(h)
        alpha = torch.softmax(router_logits, dim=-1)
        z_fused = alpha[:, 0:1] * e_w + alpha[:, 1:2] * e_s + alpha[:, 2:3] * e_t

        return {
            "shared_features": h,
            "weather_features": e_w,
            "scene_features": e_s,
            "time_features": e_t,
            "weather_logits": self.weather_head(e_w),
            "scene_logits": self.scene_head(e_s),
            "time_logits": self.time_head(e_t),
            "obj_w_logits": self.obj_head_weather(e_w),
            "obj_s_logits": self.obj_head_scene(e_s),
            "obj_t_logits": self.obj_head_time(e_t),
            "router_logits": router_logits,
            "alpha": alpha,
            "z_fused": z_fused,
            "obj_fused_logits": self.fused_obj_head(z_fused),
        }


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid], reduction="mean")


def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def router_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    mean_alpha = alpha.mean(dim=0)
    target = torch.full_like(mean_alpha, 1.0 / alpha.shape[1])
    return ((mean_alpha - target) ** 2).sum()


def expert_diversity_loss(e_w: torch.Tensor, e_s: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
    return (
        F.cosine_similarity(e_w, e_s, dim=-1).mean()
        + F.cosine_similarity(e_w, e_t, dim=-1).mean()
        + F.cosine_similarity(e_s, e_t, dim=-1).mean()
    )


@dataclass
class LossWeights:
    lambda_w_cond: float = 1.0
    lambda_s_cond: float = 1.0
    lambda_t_cond: float = 1.0
    lambda_w_obj: float = 1.0
    lambda_s_obj: float = 1.0
    lambda_t_obj: float = 1.0
    lambda_f_obj: float = 1.0
    lambda_bal: float = 0.0
    lambda_div: float = 0.0


def stage_to_loss_weights(stage: str) -> LossWeights:
    if stage == "stage0_baseline":
        return LossWeights(lambda_w_cond=0.0, lambda_s_cond=0.0, lambda_t_cond=0.0, lambda_w_obj=0.0, lambda_s_obj=0.0, lambda_t_obj=0.0, lambda_f_obj=1.0)
    if stage == "stage1_condition":
        return LossWeights(lambda_w_obj=0.2, lambda_s_obj=0.2, lambda_t_obj=0.2, lambda_f_obj=0.0)
    if stage == "stage2_experts_obj":
        return LossWeights(lambda_f_obj=0.0)
    if stage in {"stage3_moe", "stage4_finetune"}:
        return LossWeights(lambda_bal=0.01, lambda_div=0.01)
    raise ValueError(f"Unknown stage: {stage}")


def build_model(stage: str, backbone_name: str, num_weather: int, num_scene: int, num_time: int, pretrained: bool) -> nn.Module:
    if stage == "stage0_baseline":
        return PresenceBaseline(backbone_name=backbone_name, pretrained=pretrained)
    return ConditionMoEModel(backbone_name=backbone_name, num_weather=num_weather, num_scene=num_scene, num_time=num_time, pretrained=pretrained)


def maybe_compile_model(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def build_optimizer(model: nn.Module, stage: str, backbone_lr: float, expert_lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if stage == "stage0_baseline":
        param_groups = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    backbone_params = list(model.backbone.parameters())
    expert_params = list(model.weather_expert.parameters()) + list(model.scene_expert.parameters()) + list(model.time_expert.parameters()) + list(model.router.parameters())
    head_params = list(model.weather_head.parameters()) + list(model.scene_head.parameters()) + list(model.time_head.parameters()) + list(model.obj_head_weather.parameters()) + list(model.obj_head_scene.parameters()) + list(model.obj_head_time.parameters()) + list(model.fused_obj_head.parameters())

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": expert_params, "lr": expert_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, min_lr: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return output


def compute_losses(stage: str, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], pos_weight: Optional[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = stage_to_loss_weights(stage)
    objects = batch["objects"]
    loss_total = torch.tensor(0.0, device=device)
    logs: Dict[str, float] = {}

    if stage == "stage0_baseline":
        l_obj_fused = bce_logits_loss(outputs["obj_fused_logits"], objects, pos_weight=pos_weight)
        loss_total = weights.lambda_f_obj * l_obj_fused
        return loss_total, {"loss_obj_fused": float(l_obj_fused.detach().item()), "loss_total": float(loss_total.detach().item())}

    l_weather = masked_cross_entropy(outputs["weather_logits"], batch["weather_id"], batch["weather_mask"])
    l_scene = masked_cross_entropy(outputs["scene_logits"], batch["scene_id"], batch["scene_mask"])
    l_time = masked_cross_entropy(outputs["time_logits"], batch["time_id"], batch["time_mask"])

    l_obj_weather = bce_logits_loss(outputs["obj_w_logits"], objects, pos_weight=pos_weight)
    l_obj_scene = bce_logits_loss(outputs["obj_s_logits"], objects, pos_weight=pos_weight)
    l_obj_time = bce_logits_loss(outputs["obj_t_logits"], objects, pos_weight=pos_weight)

    l_obj_fused = torch.tensor(0.0, device=device)
    if weights.lambda_f_obj > 0.0:
        l_obj_fused = bce_logits_loss(outputs["obj_fused_logits"], objects, pos_weight=pos_weight)

    l_bal = router_balance_loss(outputs["alpha"]) if weights.lambda_bal > 0.0 else torch.tensor(0.0, device=device)
    l_div = expert_diversity_loss(outputs["weather_features"], outputs["scene_features"], outputs["time_features"]) if weights.lambda_div > 0.0 else torch.tensor(0.0, device=device)

    loss_total = (
        weights.lambda_w_cond * l_weather
        + weights.lambda_s_cond * l_scene
        + weights.lambda_t_cond * l_time
        + weights.lambda_w_obj * l_obj_weather
        + weights.lambda_s_obj * l_obj_scene
        + weights.lambda_t_obj * l_obj_time
        + weights.lambda_f_obj * l_obj_fused
        + weights.lambda_bal * l_bal
        + weights.lambda_div * l_div
    )

    logs.update(
        {
            "loss_weather": float(l_weather.detach().item()),
            "loss_scene": float(l_scene.detach().item()),
            "loss_time": float(l_time.detach().item()),
            "loss_obj_weather": float(l_obj_weather.detach().item()),
            "loss_obj_scene": float(l_obj_scene.detach().item()),
            "loss_obj_time": float(l_obj_time.detach().item()),
            "loss_obj_fused": float(l_obj_fused.detach().item()),
            "loss_router_balance": float(l_bal.detach().item()),
            "loss_diversity": float(l_div.detach().item()),
            "loss_total": float(loss_total.detach().item()),
        }
    )
    return loss_total, logs


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, stage: str, device: torch.device, pos_weight: Optional[torch.Tensor], num_weather: int, num_scene: int, num_time: int, amp_dtype: torch.dtype) -> Dict[str, float]:
    model.eval()

    auroc = MultilabelAUROC(num_labels=3, average="macro").to(device)
    f1 = MultilabelF1Score(num_labels=3, average="macro", threshold=0.5).to(device)
    weather_acc = MulticlassAccuracy(num_classes=num_weather).to(device)
    scene_acc = MulticlassAccuracy(num_classes=num_scene).to(device)
    time_acc = MulticlassAccuracy(num_classes=num_time).to(device)

    loss_meter: Dict[str, List[float]] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            outputs = model(batch["image"])
            _, logs = compute_losses(stage=stage, outputs=outputs, batch=batch, pos_weight=pos_weight, device=device)

        for key, value in logs.items():
            loss_meter.setdefault(key, []).append(value)

        probs = torch.sigmoid(outputs["obj_fused_logits"])
        auroc.update(probs, batch["objects"].int())
        f1.update(probs, batch["objects"].int())

        if stage != "stage0_baseline":
            weather_mask = batch["weather_mask"] > 0
            scene_mask = batch["scene_mask"] > 0
            time_mask = batch["time_mask"] > 0
            if weather_mask.any():
                weather_acc.update(outputs["weather_logits"][weather_mask], batch["weather_id"][weather_mask])
            if scene_mask.any():
                scene_acc.update(outputs["scene_logits"][scene_mask], batch["scene_id"][scene_mask])
            if time_mask.any():
                time_acc.update(outputs["time_logits"][time_mask], batch["time_id"][time_mask])

    metrics = {key: float(np.mean(values)) for key, values in loss_meter.items()}
    metrics["obj_fused_auroc"] = float(auroc.compute().item())
    metrics["obj_fused_f1"] = float(f1.compute().item())
    if stage != "stage0_baseline":
        metrics["weather_acc"] = float(weather_acc.compute().item()) if weather_acc._update_count > 0 else 0.0
        metrics["scene_acc"] = float(scene_acc.compute().item()) if scene_acc._update_count > 0 else 0.0
        metrics["time_acc"] = float(time_acc.compute().item()) if time_acc._update_count > 0 else 0.0
    return metrics


def log_epoch_metrics(prefix: str, epoch: int, metrics: Dict[str, float]) -> None:
    joined = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    print(f"[{timestamp()}] {prefix} epoch={epoch:03d} | {joined}")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, best_metric: float, config: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None, map_location: str = "cpu") -> Tuple[int, float]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", -1)), float(payload.get("best_metric", -math.inf))


def create_dataloaders(metadata_dir: Path, image_size: int, batch_size: int, num_workers: int, weighted_sampling: bool, seed: int) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    bundle = load_json(metadata_dir / "metadata_bundle.json")
    train_rows = bundle["train"]
    val_rows = bundle["val"]

    train_transform, eval_transform = build_transforms(image_size=image_size)
    train_ds = NuImagesPresenceDataset(train_rows, train_transform)
    val_ds = NuImagesPresenceDataset(val_rows, eval_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = None
    shuffle = True
    if weighted_sampling:
        sample_weights = compute_sample_weights(train_rows)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)
        shuffle = False

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(train_ds, shuffle=shuffle if train_sampler is None else False, sampler=train_sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, bundle


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, stage: str, device: torch.device, pos_weight: Optional[torch.Tensor], amp_dtype: torch.dtype, grad_clip_norm: float) -> Dict[str, float]:
    model.train()
    meter: Dict[str, List[float]] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            outputs = model(batch["image"])
            loss, logs = compute_losses(stage=stage, outputs=outputs, batch=batch, pos_weight=pos_weight, device=device)

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        for key, value in logs.items():
            meter.setdefault(key, []).append(value)

    return {key: float(np.mean(values)) for key, values in meter.items()}


def parse_amp_dtype(value: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[value]


def prepare_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    build_metadata(
        data_root=Path(args.data_root),
        train_version_root=Path(args.data_root) / args.train_version,
        val_version_root=Path(args.data_root) / args.val_version,
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    experiment_dir = Path(args.experiment_dir)
    ckpt_dir = experiment_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = parse_amp_dtype(args.amp_dtype)

    train_loader, val_loader, bundle = create_dataloaders(
        metadata_dir=Path(args.metadata_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted_sampling=args.weighted_sampling,
        seed=args.seed,
    )

    num_weather = len(bundle["weather_to_id"])
    num_scene = len(bundle["scene_to_id"])
    num_time = len(bundle["time_to_id"])

    model = build_model(
        stage=args.stage,
        backbone_name=args.backbone_name,
        num_weather=num_weather,
        num_scene=num_scene,
        num_time=num_time,
        pretrained=not args.no_pretrained,
    )
    model = maybe_compile_model(model, enabled=args.compile)
    model = model.to(device)

    optimizer = build_optimizer(model=model, stage=args.stage, backbone_lr=args.backbone_lr, expert_lr=args.expert_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_epochs=args.epochs, min_lr=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_metric = -math.inf
    epochs_without_improvement = 0

    if args.resume:
        start_epoch, best_metric = load_checkpoint(Path(args.resume), model=model, optimizer=optimizer, scheduler=scheduler, map_location="cpu")
        start_epoch += 1
        print(f"[{timestamp()}] Resumed from epoch {start_epoch} with best metric {best_metric:.4f}")

    pos_weight = compute_object_pos_weight(bundle["train"]).to(device)

    config = {key: value for key, value in vars(args).items() if key != "func"}
    config["device"] = str(device)
    config["weather_to_id"] = bundle["weather_to_id"]
    config["scene_to_id"] = bundle["scene_to_id"]
    config["time_to_id"] = bundle["time_to_id"]
    config["condition_semantics"] = bundle.get("stats", {}).get("mapping", {}).get("condition_semantics", {})
    config["object_semantics"] = bundle.get("stats", {}).get("mapping", {}).get("object_semantics", {})
    save_json(config, experiment_dir / "config.json")

    history: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            stage=args.stage,
            device=device,
            pos_weight=pos_weight,
            amp_dtype=amp_dtype,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            stage=args.stage,
            device=device,
            pos_weight=pos_weight,
            num_weather=num_weather,
            num_scene=num_scene,
            num_time=num_time,
            amp_dtype=amp_dtype,
        )

        scheduler.step()
        log_epoch_metrics("train", epoch, train_metrics)
        log_epoch_metrics("val", epoch, val_metrics)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": [group["lr"] for group in optimizer.param_groups]}
        history.append(record)
        save_json(history, experiment_dir / "history.json")

        monitor = val_metrics.get("obj_fused_auroc", -math.inf)
        improved = monitor > (best_metric + args.early_stopping_min_delta)

        if improved:
            best_metric = monitor
            epochs_without_improvement = 0
            save_checkpoint(ckpt_dir / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_metric, config=config)
            print(f"[{timestamp()}] New best model at epoch={epoch:03d} | val_obj_fused_auroc={best_metric:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"[{timestamp()}] No validation improvement for {epochs_without_improvement} epoch(s)")

        save_checkpoint(ckpt_dir / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_metric, config=config)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"[{timestamp()}] Early stopping triggered at epoch={epoch:03d} | best_val_obj_fused_auroc={best_metric:.4f}")
            break

    print(f"[{timestamp()}] Training finished. Best validation obj_fused_auroc={best_metric:.4f}")
    print(f"[{timestamp()}] Saved checkpoints: {ckpt_dir / 'best.pt'} and {ckpt_dir / 'last.pt'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nuImages Condition-Specialized MoE train/validation script")
    subparsers = parser.add_subparsers(dest="command", required=False)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare fixed JSON metadata splits from nuImages")
    prepare_parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    prepare_parser.add_argument("--train-version", type=str, default=DEFAULT_TRAIN_VERSION)
    prepare_parser.add_argument("--val-version", type=str, default=DEFAULT_VAL_VERSION)
    prepare_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.10)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(func=prepare_main)

    train_parser = subparsers.add_parser("train", help="Train the MoE model")
    train_parser.add_argument("--metadata-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--experiment-dir", type=str, default=DEFAULT_EXPERIMENT_DIR)
    train_parser.add_argument("--stage", type=str, default="stage3_moe", choices=["stage0_baseline", "stage1_condition", "stage2_experts_obj", "stage3_moe", "stage4_finetune"])
    train_parser.add_argument("--backbone-name", type=str, default="convnextv2_tiny.fcmae_ft_in1k")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--weighted-sampling", action="store_true")
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    train_parser.add_argument("--backbone-lr", type=float, default=2e-5)
    train_parser.add_argument("--expert-lr", type=float, default=1e-4)
    train_parser.add_argument("--head-lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--min-lr", type=float, default=1e-6)
    train_parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--resume", type=str, default="")
    train_parser.add_argument("--early-stopping-patience", type=int, default=10)
    train_parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    train_parser.set_defaults(func=train_main)

    return parser


def main() -> None:
    parser = build_argparser()

    if len(sys.argv) == 1:
        metadata_bundle = Path(DEFAULT_OUTPUT_DIR) / "metadata_bundle.json"
        if not metadata_bundle.exists():
            raise SystemExit(
                f"Missing metadata file: {metadata_bundle}\n"
                f"Run this first:\npython {Path(__file__).resolve()} prepare"
            )
        args = parser.parse_args(["train"])
    else:
        args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
