#!/usr/bin/env python3
"""Reconstruct the published image splits without redistributing annotations.

The reference files contain ordered image identifiers and aggregate SHA-256
checksums. Labels come from the user's licensed dataset or existing metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Dict, List, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test_fixed")
MAPPING_KEYS = ("weather_to_id", "scene_to_id", "time_to_id")
ROW_KEYS = (
    "split", "weather_id", "scene_id", "time_id", "weather_mask", "scene_mask",
    "time_mask", "car_present", "pedestrian_present", "traffic_sign_present",
)
OUTPUT_FILES = tuple(f"{split}.json" for split in SPLITS) + (
    "metadata_bundle.json", "stats.json", "preparation_audit.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_id(row: Mapping[str, Any], dataset: str) -> str:
    if row.get("image_id"):
        value = str(row["image_id"])
    else:
        path = str(row["image_path"]).replace("\\", "/")
        marker = "/images/100k/" if dataset == "bdd100k" else "/samples/"
        if marker not in path:
            raise ValueError(f"Cannot derive a stable {dataset} image ID from {path!r}")
        value = dataset + "/" + path.split(marker, 1)[1]
    parts = PurePosixPath(value).parts
    if not value.startswith(dataset + "/") or len(parts) < 3 or ".." in parts:
        raise ValueError(f"Invalid image ID {value!r}")
    return value


def canonical_row(row: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    result = {"image_id": image_id(row, dataset), "split": str(row["split"])}
    for key in ROW_KEYS[1:]:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Expected an integer for {key} in {result['image_id']}")
        if (key.endswith("_mask") or key.endswith("_present")) and value not in (0, 1):
            raise ValueError(f"Expected a binary {key} in {result['image_id']}")
        result[key] = value
    return result


def load_source(directory: Path, dataset: str):
    bundle = read_json(directory / "metadata_bundle.json")
    mappings = {key: bundle[key] for key in MAPPING_KEYS}
    rows = {split: read_json(directory / f"{split}.json") for split in SPLITS}
    canonical = {split: [canonical_row(row, dataset) for row in values] for split, values in rows.items()}
    for split, values in canonical.items():
        identities = [row["image_id"] for row in values]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate image IDs in source {split}")
    return mappings, canonical


def check_destination(directory: Path, filenames, overwrite: bool) -> None:
    existing = [name for name in filenames if (directory / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace {directory}: {existing}. Use --overwrite to replace them.")


def export_reference(args) -> None:
    destination = args.output_dir or REPOSITORY_ROOT / "data/splits" / args.dataset
    check_destination(destination, [f"{key}.json" for key in SPLITS] + ["reference.json"], args.overwrite)
    mappings, splits = load_source(args.metadata_dir, args.dataset)
    reference = {"schema_version": 1, "dataset": args.dataset, "mappings": mappings, "splits": {}}
    for split, rows in splits.items():
        identities = [row["image_id"] for row in rows]
        reference["splits"][split] = {
            "n_images": len(rows), "ordered_ids_sha256": digest(identities),
            "canonical_labels_sha256": digest(rows),
        }
        write_json(destination / f"{split}.json", identities)
    write_json(destination / "reference.json", reference)
    print(f"Exported identifier-only references to {destination}")


def load_reference(directory: Path, dataset: str):
    reference = read_json(directory / "reference.json")
    if reference.get("schema_version") != 1 or reference.get("dataset") != dataset:
        raise ValueError("Reference schema or dataset does not match this invocation")
    split_ids = {split: read_json(directory / f"{split}.json") for split in SPLITS}
    for split, ids in split_ids.items():
        expected = reference["splits"][split]
        if len(ids) != expected["n_images"] or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid image count or duplicate IDs in reference {split}")
        if digest(ids) != expected["ordered_ids_sha256"]:
            raise ValueError(f"Identifier checksum mismatch for {split}")
        for identity in ids:
            image_id({"image_id": identity}, dataset)
    return reference, split_ids


def relocated_path(identity: str, dataset: str, root: Path) -> Path:
    identity = image_id({"image_id": identity}, dataset)
    suffix = PurePosixPath(identity).relative_to(dataset)
    prefix = Path("images/100k") if dataset == "bdd100k" else Path("samples")
    return root / prefix / suffix


def load_baseline(dataset: str):
    path = REPOSITORY_ROOT / "baseline" / f"{dataset}_moe_train.py"
    name = f"release_metadata_{dataset}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import metadata functions from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows_from_annotations(args, mappings):
    baseline = load_baseline(args.dataset)
    if args.dataset == "bdd100k":
        if args.bdd_train_labels is None or args.bdd_val_labels is None:
            raise ValueError("BDD100K reconstruction requires --bdd-train-labels and --bdd-val-labels")
        rows = []
        for split, path in (("train", args.bdd_train_labels), ("val", args.bdd_val_labels)):
            for frame in read_json(path):
                row = baseline.frame_to_row(
                    frame=frame, split=split, data_root=args.image_root,
                    weather_to_id=mappings["weather_to_id"], scene_to_id=mappings["scene_to_id"],
                    time_to_id=mappings["time_to_id"],
                )
                if row is not None:
                    rows.append(asdict(row))
        return rows
    rows = []
    for split, version in (("train", args.nu_train_version), ("val", args.nu_val_version)):
        version_root = Path(version)
        if not version_root.is_absolute():
            version_root = args.image_root / version_root
        rows.extend(baseline.build_rows_for_version(
            data_root=args.image_root, version_root=version_root, split_name=split,
            location_to_id=mappings["weather_to_id"], illumination_to_id=mappings["scene_to_id"],
            motion_to_id=mappings["time_to_id"],
        ))
    return rows


def verify_and_order(rows, split_ids, reference, dataset):
    index = {}
    for row in rows:
        canonical = canonical_row(row, dataset)
        identity = canonical["image_id"]
        if identity in index and index[identity] != canonical:
            raise ValueError(f"Conflicting annotations for {identity}")
        index[identity] = canonical
    output = {}
    for split, identities in split_ids.items():
        missing = [identity for identity in identities if identity not in index]
        if missing:
            raise ValueError(f"{len(missing)} reference images lack annotations in {split}; examples {missing[:5]}")
        ordered = [dict(index[identity]) for identity in identities]
        if digest(ordered) != reference["splits"][split]["canonical_labels_sha256"]:
            raise ValueError(
                f"Label checksum mismatch for {split}. The annotations or label derivation differ from the experiment. "
                "Do not replace the release references to bypass this check."
            )
        output[split] = ordered
    return output


def split_stats(rows):
    stats = {"num_samples": len(rows), "objects": {}, "masks": {}}
    for name in ("car", "pedestrian", "traffic_sign"):
        positives = sum(row[f"{name}_present"] for row in rows)
        stats["objects"][name] = {"positive": positives, "negative": len(rows) - positives,
                                   "positive_ratio": positives / max(1, len(rows))}
    for name in ("weather", "scene", "time"):
        valid = sum(row[f"{name}_mask"] for row in rows)
        stats["masks"][name] = {"valid": valid, "invalid": len(rows) - valid,
                                 "valid_ratio": valid / max(1, len(rows))}
    return stats


def prepare(args) -> None:
    output = args.output_dir or REPOSITORY_ROOT / "data/metadata" / args.dataset
    reference_dir = args.reference_dir or REPOSITORY_ROOT / "data/splits" / args.dataset
    check_destination(output, OUTPUT_FILES, args.overwrite)
    reference, split_ids = load_reference(reference_dir, args.dataset)
    args.image_root = args.image_root.expanduser().resolve()
    missing = [identity for identity in sorted(set().union(*map(set, split_ids.values())))
               if not relocated_path(identity, args.dataset, args.image_root).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} required images are missing under {args.image_root}; examples {missing[:5]}")
    if args.existing_metadata:
        mappings, existing = load_source(args.existing_metadata, args.dataset)
        if mappings != reference["mappings"]:
            raise ValueError("Existing metadata mappings differ from the release references")
        for split in SPLITS:
            if [row["image_id"] for row in existing[split]] != split_ids[split]:
                raise ValueError(f"Existing metadata split/order differs from the release reference for {split}")
        rows = [row for values in existing.values() for row in values]
    else:
        rows = rows_from_annotations(args, reference["mappings"])
    splits = verify_and_order(rows, split_ids, reference, args.dataset)
    for values in splits.values():
        for row in values:
            row["image_path"] = str(relocated_path(row["image_id"], args.dataset, args.image_root))
    stats = {"test" if split == "test_fixed" else split: split_stats(rows) for split, rows in splits.items()}
    stats["mapping"] = dict(reference["mappings"])
    if args.dataset == "nuimages":
        stats["mapping"]["condition_semantics"] = {"weather": "location", "scene": "illumination", "time": "motion"}
        stats["mapping"]["object_semantics"] = {"slot_0": "car", "slot_1": "pedestrian", "slot_2": "traffic_cone"}
    bundle = dict(reference["mappings"])
    bundle.update({"test" if split == "test_fixed" else split: rows for split, rows in splits.items()})
    bundle["stats"] = stats
    ids = {split: set(values) for split, values in split_ids.items()}
    overlaps = {f"{left}__{right}": len(ids[left] & ids[right]) for left, right in
                (("train", "val"), ("train", "test_fixed"), ("val", "test_fixed"))}
    audit = {"dataset": args.dataset, "ordered_split_ids_verified": True, "label_checksums_verified": True,
             "image_paths_verified": True, "sizes": {key: len(value) for key, value in splits.items()},
             "overlap": overlaps, "disjoint_validation_size": len(ids["val"] - ids["train"] - ids["test_fixed"]),
             "note": ("Historical BDD100K validation/test overlap is preserved here. Revised training removes that overlap."
                      if args.dataset == "bdd100k" else "The nuImages training, validation, and fixed test pools are disjoint.")}
    for split, rows in splits.items():
        write_json(output / f"{split}.json", rows)
    write_json(output / "metadata_bundle.json", bundle)
    write_json(output / "stats.json", stats)
    write_json(output / "preparation_audit.json", audit)
    print(json.dumps(audit, indent=2))
    print(f"Prepared metadata in {output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-reference", help="Export ordered IDs and label digests from experiment metadata")
    export.add_argument("--dataset", required=True, choices=("bdd100k", "nuimages"))
    export.add_argument("--metadata-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=export_reference)
    command = commands.add_parser("prepare", help="Reconstruct metadata and verify the released splits and labels")
    command.add_argument("--dataset", required=True, choices=("bdd100k", "nuimages"))
    command.add_argument("--image-root", type=Path, required=True,
                         help="BDD100K root containing images/100k, or nuImages root containing samples")
    command.add_argument("--existing-metadata", type=Path,
                         help="Relocate existing experiment metadata instead of reading raw annotations")
    command.add_argument("--reference-dir", type=Path)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--bdd-train-labels", type=Path)
    command.add_argument("--bdd-val-labels", type=Path)
    command.add_argument("--nu-train-version", default="v1.0-train")
    command.add_argument("--nu-val-version", default="v1.0-val")
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(func=prepare)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
