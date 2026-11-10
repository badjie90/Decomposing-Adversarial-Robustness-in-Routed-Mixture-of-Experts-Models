"""Small, explicit I/O helpers for reproducible experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def save_json(path: str | Path, value: Mapping | Sequence | Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical_float(value: float) -> str:
    """Filesystem-safe representation which round-trips the supplied budget."""
    return f"{float(value):.8f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def run_stem(dataset: str, model: str, attack: str, target: str, epsilon: float, seed: int,
             acquisition_n: int | None = None) -> str:
    result = f"{dataset}__{model}__{attack}__target-{target}__eps-{canonical_float(epsilon)}__seed-{int(seed)}"
    return result if acquisition_n is None else result + f"__N-{int(acquisition_n)}"


def save_npz_and_metadata(path: str | Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], *, overwrite: bool = False) -> tuple[Path, Path]:
    """Atomically avoid accidental overwrite of a costly experimental run."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not overwrite and (path.exists() or meta_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing run artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Supplying a file handle prevents numpy from silently appending .npz
        # to the temporary filename.  The final replace is atomic on this
        # filesystem, so disk-full failures cannot masquerade as a completed
        # raw artifact at the expected path.
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    save_json(meta_path, metadata)
    return path, meta_path


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
