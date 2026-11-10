#!/usr/bin/env python3
"""Generate and execute restartable, GPU-pinned experiment job queues."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .io import run_stem
from .paths import csa_run, metadata_dir, output_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = output_dir()
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_EPS = 8 / 255
SENSITIVITY_EPS = (2 / 255, 4 / 255, 8 / 255, .09)
ACQUISITION_SIZES = (50, 100, 250, 500, 1000, 2000)


def rows_for(phase: str, datasets: list[str], models: list[str], epsilons: list[float], seeds: list[int]):
    rows = []
    for dataset in datasets:
        for model in models:
            if phase in ("primary", "sparse-primary"):
                for attack in ("square", "hopskipjump"):
                    for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="attack", attack=attack, target="router", seed=seed, epsilon=PRIMARY_EPS, acquisition_n=""))
                for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="transfer", attack="transfer_pgd", target="router", seed=seed, epsilon=PRIMARY_EPS, acquisition_n=2000))
            elif phase in ("controls", "sparse-controls"):
                for attack in ("square", "hopskipjump"):
                    for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="attack", attack=attack, target="fused", seed=seed, epsilon=PRIMARY_EPS, acquisition_n=""))
                for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="transfer", attack="transfer_pgd", target="fused", seed=seed, epsilon=PRIMARY_EPS, acquisition_n=2000))
            elif phase == "sensitivity":
                for epsilon in epsilons:
                    for target in ("router", "fused"):
                        for attack in ("square", "hopskipjump"):
                            for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="attack", attack=attack, target=target, seed=seed, epsilon=epsilon, acquisition_n=""))
                        for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="transfer", attack="transfer_pgd", target=target, seed=seed, epsilon=epsilon, acquisition_n=2000))
            elif phase == "transfer-scaling":
                for target in ("router", "fused"):
                    for n in ACQUISITION_SIZES:
                        for seed in seeds: rows.append(dict(dataset=dataset, model=model, kind="transfer", attack="transfer_pgd", target=target, seed=seed, epsilon=PRIMARY_EPS, acquisition_n=n))
            else:
                raise ValueError(phase)
    return rows


def model_args(row, output: Path) -> list[str]:
    if row["model"] == "csa":
        run = csa_run(row["dataset"], output)
        return ["--model", "csa", "--run-dir", str(run)]
    return ["--model", "sparse", "--sparse-checkpoint", str(output / "models" / row["dataset"] / "sparse_seed42" / "checkpoints" / "best.pt")]


def expected_path(row, output: Path) -> Path:
    n = int(row["acquisition_n"]) if row["kind"] == "transfer" else None
    # ``revision.run attack`` deliberately exposes the familiar
    # ``hopskipjump`` CLI spelling but records the canonical short name
    # ``hsj`` in raw artifact names.  Use the same name here so a completed
    # HSJ job is recognised on a restart instead of being run again (and then
    # failing when the immutable raw artifact already exists).
    attack = "hsj" if row["attack"] == "hopskipjump" else row["attack"]
    stem = run_stem(row["dataset"], row["model"], attack, row["target"], float(row["epsilon"]), int(row["seed"]), n)
    return output / "raw_runs" / (stem + ".npz")


def _snapshot_paths(row: Mapping[str, str], output: Path, query_checkpoints: Sequence[int]) -> list[Path]:
    """Return the persisted query snapshots for a black-box attack job."""
    if row["kind"] != "attack":
        return []
    attack = "hsj" if row["attack"] == "hopskipjump" else row["attack"]
    stem = run_stem(row["dataset"], row["model"], attack, row["target"], float(row["epsilon"]), int(row["seed"]))
    result = []
    for query in query_checkpoints:
        path = output / "query_budgets" / f"{stem}__Q-{int(query)}.npz"
        result.extend((path, path.with_suffix(".json")))
    return result


def job_artifact_paths(row: Mapping[str, str], output: Path, query_checkpoints: Sequence[int]) -> list[Path]:
    """All files required for a run to be considered complete and reportable."""
    raw = expected_path(row, output)
    return [raw, raw.with_suffix(".json"), raw.with_suffix(".csv"), *_snapshot_paths(row, output, query_checkpoints)]


def _is_valid_npz(path: Path) -> bool:
    """Reject incomplete archives without materialising their large tensors."""
    if not (path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)):
        return False
    try:
        # Opening an ``.npz`` verifies its members' checksums.  Only
        # ``image_ids`` is decompressed, so this remains cheap for the large
        # feature tensors in a normal raw artifact.
        with np.load(path, allow_pickle=False) as archive:
            return "image_ids" in archive.files and archive["image_ids"].size > 0
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _canonical_attack(row: Mapping[str, str]) -> str:
    return "hsj" if row["attack"] == "hopskipjump" else row["attack"]


def _metadata_matches(path: Path, row: Mapping[str, str], *, snapshot_query: int | None = None) -> bool:
    """Check the identity fields that tie an artifact to its queue row."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        identity_matches = (
            metadata["dataset"] == row["dataset"]
            and metadata["model"] == row["model"]
            and metadata["attack"] == _canonical_attack(row)
            and metadata["target"] == row["target"]
            and int(metadata["seed"]) == int(row["seed"])
            and math.isclose(float(metadata["epsilon"]), float(row["epsilon"]), rel_tol=0.0, abs_tol=1e-12)
        )
        if snapshot_query is not None:
            identity_matches = identity_matches and int(metadata["requested_query_budget"]) == int(snapshot_query)
        return bool(identity_matches)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _has_image_rows(path: Path) -> bool:
    """A reportable per-image CSV needs an image-id column and one row."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return bool(rows.fieldnames and "image_id" in rows.fieldnames and next(rows, None) is not None)
    except (OSError, csv.Error, UnicodeDecodeError):
        return False


def job_is_complete(row: Mapping[str, str], output: Path, query_checkpoints: Sequence[int]) -> bool:
    """Require a complete, readable artifact family before a worker skips it."""
    raw = expected_path(row, output)
    if not _is_valid_npz(raw):
        return False
    if not _metadata_matches(raw.with_suffix(".json"), row):
        return False
    if not _has_image_rows(raw.with_suffix(".csv")):
        return False
    for query in query_checkpoints if row["kind"] == "attack" else ():
        snapshot = output / "query_budgets" / f"{raw.stem}__Q-{int(query)}.npz"
        if not _is_valid_npz(snapshot):
            return False
        if not _metadata_matches(snapshot.with_suffix(".json"), row, snapshot_query=int(query)):
            return False
    return True


def _transfer_surrogate_path(row: Mapping[str, str], output: Path) -> Path | None:
    if row["kind"] != "transfer":
        return None
    stem = run_stem(row["dataset"], row["model"], "transfer_pgd", row["target"], float(row["epsilon"]),
                    int(row["seed"]), int(row["acquisition_n"]))
    return output / "surrogates" / f"{stem}.pt"


def quarantine_incomplete_artifacts(row: Mapping[str, str], output: Path, query_checkpoints: Sequence[int]) -> Path | None:
    """Move partial outputs aside before retrying a job.

    This is deliberately a same-filesystem rename, so it is reversible and
    consumes no duplicate disk space.  A transfer surrogate is included only
    when its paired raw result is incomplete; otherwise it remains reusable.
    """
    paths = job_artifact_paths(row, output, query_checkpoints)
    surrogate = _transfer_surrogate_path(row, output)
    if surrogate is not None:
        paths.append(surrogate)
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    expected = expected_path(row, output)
    destination_root = output / "failed_artifacts" / f"{expected.stem}__{time.time_ns()}"
    for source in existing:
        destination = destination_root / source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    return destination_root


def cmd_generate(args):
    eps = [float(x) for x in args.epsilons] if args.epsilons else list(SENSITIVITY_EPS if args.phase == "sensitivity" else (PRIMARY_EPS,))
    rows = rows_for(args.phase, args.datasets, args.models, eps, args.seeds)
    path = args.jobs_file.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "model", "kind", "attack", "target", "seed", "epsilon", "acquisition_n"), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} jobs to {path}")


def cmd_worker(args):
    output = args.output_dir.resolve(); jobs = list(csv.DictReader(args.jobs_file.open(encoding="utf-8"), delimiter="\t"))
    assigned = [(i, row) for i, row in enumerate(jobs) if i % args.workers == args.worker_index]
    env = dict(os.environ, PYTHONPATH=str(ROOT) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""), MPLCONFIGDIR=str(ROOT / ".mplconfig"), CUDA_VISIBLE_DEVICES=str(args.gpu))
    log_dir = output / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, row in assigned:
        expected = expected_path(row, output)
        if job_is_complete(row, output, args.query_checkpoints):
            print(f"SKIP existing {expected.name}", flush=True); continue
        quarantined = None if args.dry_run else quarantine_incomplete_artifacts(row, output, args.query_checkpoints)
        if quarantined is not None:
            print(f"QUARANTINED incomplete artifacts: {quarantined}", flush=True)
        # ``revision.run`` puts options on individual subparsers, so the
        # subcommand must appear before every job option.
        if row["kind"] == "attack":
            command = [sys.executable, "-m", "revision.run", "attack", "--output-dir", str(output), "--device", "cuda:0", "--dataset", row["dataset"], *model_args(row, output), "--metadata-dir", str(metadata_dir(row["dataset"])), "--manifest", str(output / "manifests" / f"{row['dataset']}_n1000_seed20261029.json"), "--seed", row["seed"], "--target", row["target"], "--epsilon", row["epsilon"], "--attack", row["attack"], "--query-budget", str(args.query_budget), "--query-checkpoints", *map(str, args.query_checkpoints), "--progress-every", str(args.progress_every)]
        else:
            command = [sys.executable, "-m", "revision.run", "transfer", "--output-dir", str(output), "--device", "cuda:0", "--dataset", row["dataset"], *model_args(row, output), "--metadata-dir", str(metadata_dir(row["dataset"])), "--manifest", str(output / "manifests" / f"{row['dataset']}_n1000_seed20261029.json"), "--seed", row["seed"], "--target", row["target"], "--epsilon", row["epsilon"], "--acquisition-n", row["acquisition_n"], "--bank-max", str(args.bank_max), "--batch-size", str(args.batch_size), "--surrogate-epochs", str(args.surrogate_epochs), "--pgd-steps", str(args.pgd_steps), "--pgd-restarts", str(args.pgd_restarts), "--progress-every", str(args.progress_every)]
        log = log_dir / (expected.stem + ".log")
        print(f"START worker={args.worker_index} GPU={args.gpu}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"FAILED ({result.returncode}): {log}", flush=True)
            if not args.continue_on_error: raise SystemExit(result.returncode)
        else:
            print(f"DONE: {expected.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate"); generate.add_argument("--phase", choices=("primary", "controls", "sensitivity", "transfer-scaling", "sparse-primary", "sparse-controls"), required=True); generate.add_argument("--jobs-file", type=Path, required=True); generate.add_argument("--datasets", nargs="+", default=("bdd100k", "nuimages")); generate.add_argument("--models", nargs="+", default=("csa",)); generate.add_argument("--seeds", type=int, nargs="+", default=SEEDS); generate.add_argument("--epsilons", type=float, nargs="*"); generate.set_defaults(func=cmd_generate)
    worker = sub.add_parser("worker"); worker.add_argument("--jobs-file", type=Path, required=True); worker.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); worker.add_argument("--worker-index", type=int, required=True); worker.add_argument("--workers", type=int, required=True); worker.add_argument("--gpu", required=True); worker.add_argument("--query-budget", type=int, default=1000); worker.add_argument("--query-checkpoints", type=int, nargs="+", default=(100, 250, 500, 1000)); worker.add_argument("--bank-max", type=int, default=2000); worker.add_argument("--batch-size", type=int, default=32); worker.add_argument("--surrogate-epochs", type=int, default=20); worker.add_argument("--pgd-steps", type=int, default=100); worker.add_argument("--pgd-restarts", type=int, default=3); worker.add_argument("--progress-every", type=int, default=10); worker.add_argument("--continue-on-error", action="store_true"); worker.add_argument("--dry-run", action="store_true"); worker.set_defaults(func=cmd_worker)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
