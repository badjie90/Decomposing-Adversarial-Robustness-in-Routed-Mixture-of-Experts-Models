"""Stable image identities, deterministic pixel preprocessing and split audits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .paths import image_root
_SLOTS = ("weather", "scene", "time")


def class_names_for(dataset: str):
    if dataset.lower() == "bdd100k":
        return ["car", "pedestrian", "traffic_sign"]
    if dataset.lower() == "nuimages":
        return ["car", "pedestrian", "traffic_cone"]
    raise ValueError(f"Unknown dataset {dataset!r}")


def expert_names_for(dataset: str):
    class_names_for(dataset)
    return ["weather", "scene", "time"] if dataset.lower() == "bdd100k" else ["location", "illumination", "motion"]


def stable_image_id(row: Mapping[str, Any], dataset: str) -> str:
    """Location-independent identity retains split or camera directory."""
    if row.get("image_id"):
        return str(row["image_id"])
    path = str(row["image_path"]).replace("\\", "/")
    dataset = dataset.lower()
    marker = "/images/100k/" if dataset == "bdd100k" else "/samples/"
    if marker in path:
        return dataset + "/" + path.split(marker, 1)[1]
    return dataset + "/" + str(row.get("split", "unknown")) + "/" + Path(path).name


def load_metadata(dataset: str, metadata_dir: str, split: Optional[str] = "test_fixed") -> Tuple[list, dict]:
    class_names_for(dataset)
    directory = Path(metadata_dir).expanduser().resolve()
    full = json.loads((directory / "metadata_bundle.json").read_text())
    # The historical bundle embeds all three full image lists. Avoid copying
    # hundreds of thousands of records into every experiment's manifest.
    bundle = {key: value for key, value in full.items() if key not in ("train", "val", "test", "test_fixed")}
    bundle["metadata_dir"] = str(directory)
    if split is None:
        return [], bundle
    filename = "test_fixed" if split == "test" else split
    rows = json.loads((directory / f"{filename}.json").read_text())
    for row in rows:
        row["image_id"] = stable_image_id(row, dataset)
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate image IDs in {directory / (filename + '.json')}")
    return rows, bundle


def resolve_image_path(path: str, dataset: str, image_roots: Optional[Sequence[str]] = None) -> Path:
    """Resolve only explicit historical moves or dataset-relative candidates.

    Missing or ambiguous files fail; an unrelated image with the same basename
    is never silently substituted through a recursive filesystem search.
    Explicit roots should contain images/100k or samples, or be those folders.
    """
    original = Path(path).expanduser()
    if original.is_file():
        return original.resolve()
    dataset = dataset.lower()
    if dataset not in ("bdd100k", "nuimages"):
        raise ValueError(f"Unknown dataset {dataset!r}")
    relative_dir = "images/100k" if dataset == "bdd100k" else "samples"
    marker = "/" + relative_dir + "/"
    normalized = str(path).replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
    elif normalized.startswith(relative_dir + "/"):
        suffix = normalized[len(relative_dir) + 1:]
    elif normalized.startswith(dataset + "/"):
        suffix = normalized[len(dataset) + 1:]
    else:
        suffix = normalized if not Path(normalized).is_absolute() else None
    candidates = []
    for root in [*list(image_roots or []), image_root(dataset)]:
        root = Path(root).expanduser()
        if suffix:
            candidates.extend((root / relative_dir / suffix, root / suffix))
    matches = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})
    if len(matches) > 1:
        raise ValueError(f"Ambiguous relocated image {path}: {matches}")
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing {dataset} image: {path}; checked explicit roots {image_roots or []}")


def row_context(row: Mapping[str, Any], dataset: str, bundle: Mapping[str, Any]) -> Dict[str, str]:
    result = {}
    for slot, semantic in zip(_SLOTS, expert_names_for(dataset)):
        mapping = bundle.get(f"{slot}_to_id", {})
        lookup = {int(value): key for key, value in mapping.items()}
        valid = int(row.get(f"{slot}_mask", 0)) == 1
        label_id = int(row.get(f"{slot}_id", -1))
        result[semantic] = str(lookup.get(label_id, label_id)) if valid else "unknown"
    return result


class PixelDataset(Dataset):
    """The original evaluation resize/center crop, without normalization.

    Each item: image [3,H,W] float32 in [0,1], labels [3] float32,
    image_id str, image_path str, contexts {semantic_name: label_name}.
    Historical nuImages traffic_sign_present is the TRAFFIC CONE label slot.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], image_size: int, dataset: str,
                 metadata_bundle: Optional[Mapping[str, Any]] = None,
                 image_roots: Optional[Sequence[str]] = None):
        from torchvision import transforms

        self.rows = list(rows)
        self.dataset = dataset.lower()
        self.bundle = dict(metadata_bundle or {})
        self.image_roots = image_roots
        self.class_names = class_names_for(dataset)
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size), transforms.ToTensor(),
        ])
        self.image_size = int(image_size)
        self.image_ids = [stable_image_id(row, dataset) for row in self.rows]
        if len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("PixelDataset requires unique image IDs.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = resolve_image_path(row["image_path"], self.dataset, self.image_roots)
        with Image.open(path) as source:
            pixels = self.transform(source.convert("RGB"))
        labels = torch.tensor([row["car_present"], row["pedestrian_present"], row["traffic_sign_present"]], dtype=torch.float32)
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(f"Non-binary labels for {self.image_ids[index]}")
        return {"image": pixels, "labels": labels,
                "image_id": self.image_ids[index], "image_path": str(path),
                "contexts": row_context(row, self.dataset, self.bundle)}

    def validate_paths(self) -> Dict[str, Any]:
        missing = []
        for row in self.rows:
            try:
                resolve_image_path(row["image_path"], self.dataset, self.image_roots)
            except (FileNotFoundError, ValueError) as exc:
                missing.append(str(exc))
        return {"n_images": len(self), "n_missing_or_ambiguous": len(missing), "examples": missing[:10]}


def stratified_indices(routes: Sequence[int], n: int, seed: int, mode: str = "proportional",
                       image_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """Exact-size sample without replacement, with every observed route present.

    Proportional allocation uses largest remainders with a one-image minimum;
    balanced allocation fills equally until a stratum's capacity is exhausted.
    Supply image_ids for invariance to metadata-row ordering. Stratification must
    use the reference model ONCE, then reuse the stored IDs across all models.
    """
    routes = np.asarray(routes)
    if routes.ndim != 1 or not np.issubdtype(routes.dtype, np.integer) or np.any(routes < 0):
        raise ValueError("Routes must be a one-dimensional nonnegative integer array.")
    groups, counts = np.unique(routes, return_counts=True)
    if not isinstance(n, (int, np.integer)) or not len(groups) <= n <= len(routes):
        raise ValueError("Sample size must cover every observed route without replacement.")
    if image_ids is not None and (len(image_ids) != len(routes) or len(set(image_ids)) != len(image_ids)):
        raise ValueError("image_ids must be unique and match the route array.")
    if mode == "proportional":
        expected = n * counts / counts.sum()
    elif mode == "balanced":
        expected = np.full(len(groups), n / len(groups))
    else:
        raise ValueError("mode must be proportional or balanced")
    allocation = np.minimum(counts, np.maximum(1, np.floor(expected).astype(int)))
    while allocation.sum() > n:
        score = np.where(allocation > 1, allocation - expected, -np.inf)
        allocation[int(np.argmax(score))] -= 1
    while allocation.sum() < n:
        score = np.where(allocation < counts, expected - allocation, -np.inf)
        allocation[int(np.argmax(score))] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for group, count in zip(groups, allocation):
        candidates = np.flatnonzero(routes == group)
        if image_ids is not None:
            candidates = np.asarray(sorted(candidates, key=lambda i: image_ids[i]))
        selected.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
    selected = sorted(selected, key=(lambda i: image_ids[i]) if image_ids is not None else None)
    assert len(selected) == n and len(set(selected)) == n
    return np.asarray(selected, dtype=np.int64)


def split_audit(dataset: str, metadata_dir: str) -> Dict[str, Any]:
    splits = {key: load_metadata(dataset, metadata_dir, key)[0] for key in ("train", "val", "test_fixed")}
    ids = {key: {row["image_id"] for row in rows} for key, rows in splits.items()}
    intersections = {}
    for left, right in (("train", "val"), ("train", "test_fixed"), ("val", "test_fixed")):
        overlap = sorted(ids[left] & ids[right])
        intersections[f"{left}__{right}"] = {"count": len(overlap), "examples": overlap[:5]}
    return {"dataset": dataset, "sizes": {key: len(rows) for key, rows in splits.items()},
            "overlap": intersections,
            "test_disjoint_from_train_and_validation": not bool((ids["train"] | ids["val"]) & ids["test_fixed"])}


def disjoint_validation_rows(dataset: str, metadata_dir: str) -> Tuple[list, Dict[str, Any]]:
    """Explicitly remove train/test overlap for NEW model selection only.

    This does not retroactively repair historical checkpoint selection leakage.
    Save the returned audit and resulting image IDs with the new training run.
    """
    train, _ = load_metadata(dataset, metadata_dir, "train")
    val, _ = load_metadata(dataset, metadata_dir, "val")
    test, _ = load_metadata(dataset, metadata_dir, "test_fixed")
    excluded = {row["image_id"] for row in train + test}
    rows = [row for row in val if row["image_id"] not in excluded]
    if not rows:
        raise ValueError("No disjoint validation images remain.")
    return rows, {"original_validation_count": len(val), "retained_validation_count": len(rows),
                  "removed_train_or_test_overlap": len(val) - len(rows),
                  "historical_checkpoint_selection_repaired": False}
