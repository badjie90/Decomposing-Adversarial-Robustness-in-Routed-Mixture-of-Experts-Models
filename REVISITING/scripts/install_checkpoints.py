#!/usr/bin/env python3
"""Verify and install the separately distributed selected model checkpoints.

Only files named in artifacts/checkpoints.json are accepted. Archive members
are streamed to temporary files rather than extracted with tarfile.extractall.
No checkpoint deserialization or third-party packages are needed to install.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path {name!r}")
    if not name.startswith("results/models/") or str(path) != name:
        raise ValueError(f"Unexpected archive path {name!r}")
    return Path(*path.parts)


def destination_path(output: Path, name: str) -> Path:
    relative = checked_relative_path(name).relative_to("results")
    destination = output / relative
    if destination.is_symlink() or output not in destination.resolve().parents:
        raise ValueError(f"Destination must stay inside the output directory: {destination}")
    return destination


def install(bundle: Path, root: Path, verify_only: bool = False,
            output_dir: Path | None = None) -> dict:
    manifest = json.loads((REPOSITORY_ROOT / "artifacts/checkpoints.json").read_text())
    metadata = manifest["bundle"]
    if bundle.stat().st_size != metadata["bytes"]:
        raise ValueError("Checkpoint bundle size does not match artifacts/checkpoints.json")
    if sha256_file(bundle) != metadata["sha256"]:
        raise ValueError("Checkpoint bundle SHA256 does not match artifacts/checkpoints.json")
    files = {record["path"]: record for record in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Duplicate file records in checkpoint manifest")
    for name in files:
        checked_relative_path(name)
    root = root.expanduser().resolve()
    output = Path(output_dir if output_dir is not None else os.environ.get("OUTPUT_DIR", "results")).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse replacement of existing training results before writing any file.
    # Identical files are allowed, which makes a repeated install harmless.
    for name, record in files.items():
        destination = destination_path(output, name)
        if destination.exists() and (not destination.is_file() or sha256_file(destination) != record["sha256"]):
            raise FileExistsError(f"Refusing to replace a different existing file: {destination}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-install-", dir=output) as staging:
        staging_root = Path(staging)
        seen = set()
        with tarfile.open(bundle, "r:gz") as archive:
            for member in archive:
                if member.name not in files or member.name in seen or not member.isfile():
                    raise ValueError(f"Unexpected, duplicate, or nonregular archive member: {member.name}")
                record = files[member.name]
                if member.size != record["bytes"]:
                    raise ValueError(f"Archive member size mismatch: {member.name}")
                temporary = staging_root / checked_relative_path(member.name)
                temporary.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with source, temporary.open("wb") as output_stream:
                    shutil.copyfileobj(source, output_stream, length=1024 * 1024)
                if sha256_file(temporary) != record["sha256"]:
                    raise ValueError(f"Archive member SHA256 mismatch: {member.name}")
                seen.add(member.name)
        if seen != set(files):
            raise ValueError(f"Missing archive members: {sorted(set(files) - seen)}")
        if not verify_only:
            for name in sorted(files):
                destination = destination_path(output, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Same-filesystem linking publishes the verified file without
                    # replacing an output created while the archive was checked.
                    os.link(staging_root / name, destination)
                except FileExistsError:
                    if (destination.is_symlink() or not destination.is_file() or
                            sha256_file(destination) != files[name]["sha256"]):
                        raise FileExistsError(f"Refusing to replace a different existing file: {destination}")
    return {
        "status": "verified" if verify_only else "installed",
        "checkpoints": len(manifest["checkpoints"]),
        "files": len(files),
        "bundle_sha256": metadata["sha256"],
        "destination": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Local path to checkpoints.tar.gz")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT,
                        help="Repository root used to resolve a relative output directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Experiment output directory; defaults to OUTPUT_DIR or results relative to --root")
    parser.add_argument("--verify-only", action="store_true",
                        help="Validate the archive without installing its files")
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.bundle.expanduser().resolve(), args.root, args.verify_only,
                                 args.output_dir), indent=2))
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Checkpoint installation failed: {error}\n")


if __name__ == "__main__":
    main()
