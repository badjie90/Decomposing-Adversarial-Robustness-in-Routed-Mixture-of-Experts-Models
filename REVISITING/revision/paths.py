"""Repository-relative defaults with explicit environment overrides."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def output_dir() -> Path:
    return configured_path('OUTPUT_DIR', ROOT / 'results')


def metadata_dir(dataset: str) -> Path:
    key = {'bdd100k': 'BDD_METADATA', 'nuimages': 'NUIMAGES_METADATA'}[dataset]
    return configured_path(key, ROOT / 'data' / 'metadata' / dataset)


def csa_run(dataset: str, output: Path | None = None) -> Path:
    key = {'bdd100k': 'BDD_CSA_RUN', 'nuimages': 'NUIMAGES_CSA_RUN'}[dataset]
    name = 'csa_disjoint_seed42' if dataset == 'bdd100k' else 'csa_seed42'
    return configured_path(key, (output or output_dir()) / 'models' / dataset / name)


def image_root(dataset: str) -> Path:
    key = {'bdd100k': 'BDD_IMAGE_ROOT', 'nuimages': 'NUIMAGES_IMAGE_ROOT'}[dataset]
    return configured_path(key, ROOT / 'data' / dataset)
