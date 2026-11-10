"""Check exact split preservation and mismatch rejection without dataset access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data_release", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def example_row(identity, split="train"):
    return {"image_id": identity, "split": split, "weather_id": 0, "scene_id": 1,
            "time_id": 2, "weather_mask": 1, "scene_mask": 1, "time_mask": 1,
            "car_present": 1, "pedestrian_present": 0, "traffic_sign_present": 1}


def fixture_metadata(tmp_path):
    dataset = "bdd100k"
    first = example_row("bdd100k/train/a.jpg")
    second = example_row("bdd100k/train/b.jpg")
    third = example_row("bdd100k/val/c.jpg", "val")
    splits = {"train": [first], "val": [third, second], "test_fixed": [third]}
    mappings = {key: {"example": 0} for key in helper.MAPPING_KEYS}
    source = tmp_path / "source"
    for split, rows in splits.items():
        helper.write_json(source / f"{split}.json", rows)
    helper.write_json(source / "metadata_bundle.json", mappings)
    references = tmp_path / "references"
    helper.export_reference(SimpleNamespace(dataset=dataset, metadata_dir=source, output_dir=references, overwrite=False))
    return source, references, splits


def test_relocation_preserves_labels_order_and_overlap(tmp_path):
    source, references, splits = fixture_metadata(tmp_path)
    images = tmp_path / "images"
    for rows in splits.values():
        for row in rows:
            path = helper.relocated_path(row["image_id"], "bdd100k", images)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    output = tmp_path / "prepared"
    helper.prepare(SimpleNamespace(dataset="bdd100k", image_root=images, existing_metadata=source,
                                   reference_dir=references, output_dir=output, overwrite=False))
    for split, expected in splits.items():
        actual = json.loads((output / f"{split}.json").read_text())
        assert [helper.canonical_row(row, "bdd100k") for row in actual] == expected
        assert all(Path(row["image_path"]).is_file() for row in actual)
    audit = helper.read_json(output / "preparation_audit.json")
    assert audit["overlap"]["val__test_fixed"] == 1
    assert audit["disjoint_validation_size"] == 1


def test_label_change_is_rejected(tmp_path):
    _, references, splits = fixture_metadata(tmp_path)
    reference, identifiers = helper.load_reference(references, "bdd100k")
    rows = [row.copy() for values in splits.values() for row in values]
    rows[0]["car_present"] = 0
    with pytest.raises(ValueError, match="Label checksum mismatch"):
        helper.verify_and_order(rows, identifiers, reference, "bdd100k")


def test_changed_reference_order_is_rejected(tmp_path):
    _, references, _ = fixture_metadata(tmp_path)
    path = references / "val.json"
    helper.write_json(path, list(reversed(helper.read_json(path))))
    with pytest.raises(ValueError, match="Identifier checksum mismatch"):
        helper.load_reference(references, "bdd100k")


def test_path_traversal_is_rejected():
    with pytest.raises(ValueError, match="Invalid image ID"):
        helper.relocated_path("bdd100k/../../outside.jpg", "bdd100k", Path("/tmp/images"))
