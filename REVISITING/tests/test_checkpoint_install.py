"""Checkpoint installation must verify content before writing results."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "checkpoint_installer", Path(__file__).resolve().parents[1] / "scripts/install_checkpoints.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def make_bundle(tmp_path, monkeypatch, contents, records=None):
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    root = tmp_path / "source"
    (root / "artifacts").mkdir(parents=True)
    archive_path = tmp_path / "checkpoints.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    files = records if records is not None else [
        {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in contents.items()
    ]
    manifest = {
        "bundle": {"bytes": archive_path.stat().st_size, "sha256": installer.sha256_file(archive_path)},
        "files": files,
        "checkpoints": [{"id": "test-checkpoint"}],
    }
    (root / "artifacts/checkpoints.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(installer, "REPOSITORY_ROOT", root)
    return archive_path


def test_install_verify_and_repeat(tmp_path, monkeypatch):
    name = "results/models/example/checkpoints/best.pt"
    bundle = make_bundle(tmp_path, monkeypatch, {name: b"test model bytes"})
    destination = tmp_path / "installed"
    assert installer.install(bundle, destination, verify_only=True)["status"] == "verified"
    assert not (destination / name).exists()
    installer.install(bundle, destination)
    assert (destination / name).read_bytes() == b"test model bytes"
    installer.install(bundle, destination)
    (destination / name).write_bytes(b"independent local training result")
    with pytest.raises(FileExistsError):
        installer.install(bundle, destination)
    assert (destination / name).read_bytes() == b"independent local training result"


def test_reject_corrupt_bundle(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, monkeypatch, {"results/models/example/best.pt": b"weights"})
    data = bytearray(bundle.read_bytes())
    data[len(data) // 2] ^= 1
    bundle.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        installer.install(bundle, tmp_path / "installed")


def test_reject_file_mismatch_before_install(tmp_path, monkeypatch):
    name = "results/models/example/best.pt"
    record = {"path": name, "bytes": 7, "sha256": hashlib.sha256(b"correct").hexdigest()}
    bundle = make_bundle(tmp_path, monkeypatch, {name: b"changed"}, [record])
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="member SHA256"):
        installer.install(bundle, destination)
    assert not (destination / name).exists()


def test_reject_path_traversal(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, monkeypatch, {"results/models/../../outside.pt": b"weights"})
    with pytest.raises(ValueError, match="Unsafe archive path"):
        installer.install(bundle, tmp_path / "installed")


@pytest.mark.parametrize("absolute", [False, True])
def test_custom_output_and_existing_file_protection(tmp_path, monkeypatch, absolute):
    name = "results/models/example/checkpoints/best.pt"
    bundle = make_bundle(tmp_path, monkeypatch, {name: b"selected weights"})
    root = tmp_path / "checkout"
    output = tmp_path / "external storage" if absolute else Path("alternate/results")
    expected = output if absolute else root / output
    result = installer.install(bundle, root, output_dir=output)
    checkpoint = expected / "models/example/checkpoints/best.pt"
    assert result["destination"] == str(expected)
    assert checkpoint.read_bytes() == b"selected weights"
    assert not (root / "results").exists()
    installer.install(bundle, root, output_dir=output)
    checkpoint.write_bytes(b"new local training result")
    with pytest.raises(FileExistsError):
        installer.install(bundle, root, output_dir=output)
    assert checkpoint.read_bytes() == b"new local training result"


def test_environment_output_and_explicit_override(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, monkeypatch, {"results/models/example/best.pt": b"weights"})
    root = tmp_path / "checkout"
    monkeypatch.setenv("OUTPUT_DIR", "environment_outputs")
    installer.install(bundle, root)
    assert (root / "environment_outputs/models/example/best.pt").read_bytes() == b"weights"
    installer.install(bundle, root, output_dir=Path("explicit_outputs"))
    assert (root / "explicit_outputs/models/example/best.pt").read_bytes() == b"weights"


def test_reject_symlink_escape_from_custom_output(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, monkeypatch, {"results/models/example/best.pt": b"weights"})
    output = tmp_path / "chosen_output"
    external = tmp_path / "external_models"
    output.mkdir()
    external.mkdir()
    (output / "models").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the output directory"):
        installer.install(bundle, tmp_path / "checkout", output_dir=output)
    assert not list(external.iterdir())
