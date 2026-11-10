"""Keep local configuration from redirecting a separate reproduction run."""
import os
from pathlib import Path
import shutil
import subprocess


def _repository(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/common.sh"
    shutil.copyfile(source, root / "scripts/common.sh")
    (root / "configs/paths.env").write_text(
        'OUTPUT_DIR="configured results"\nBDD_METADATA="data/custom metadata"\n'
    )
    environment = dict(os.environ)
    for name in ("OUTPUT_DIR", "PYTHON_BIN", "BDD_METADATA", "NUIMAGES_METADATA",
                 "BDD_IMAGE_ROOT", "NUIMAGES_IMAGE_ROOT", "BDD_CSA_RUN",
                 "NUIMAGES_CSA_RUN", "MPLCONFIGDIR"):
        environment.pop(name, None)
    return root, environment


def test_caller_output_overrides_local_config(tmp_path):
    root, environment = _repository(tmp_path)
    environment["OUTPUT_DIR"] = "separate results"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$OUTPUT_DIR" "$BDD_METADATA"',
         "test", str(root / "scripts/common.sh")],
        env=environment, text=True, capture_output=True, check=True,
    )
    assert result.stdout.splitlines() == [str(root / "separate results"), "data/custom metadata"]


def test_child_run_recomputes_model_defaults(tmp_path):
    root, environment = _repository(tmp_path)
    command = '''source "$1"
OUTPUT_DIR="retrained results" bash -c 'source "$1"; printf "%s\\n" "$BDD_CSA_RUN" "$NUIMAGES_CSA_RUN"' test "$1"
'''
    result = subprocess.run(
        ["bash", "-c", command, "test", str(root / "scripts/common.sh")],
        env=environment, text=True, capture_output=True, check=True,
    )
    assert result.stdout.splitlines() == [
        str(root / "retrained results/models/bdd100k/csa_disjoint_seed42"),
        str(root / "retrained results/models/nuimages/csa_seed42"),
    ]
