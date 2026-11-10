# Dataset preparation

The repository includes ordered image identifiers for the experiment splits. It does not include images or object annotations. Obtain BDD100K and nuImages through their dataset providers and follow their access and licence terms.

`scripts/prepare_data.py` derives labels from the original annotation files, selects the released image identifiers, and preserves their order. It verifies SHA-256 checksums of the ordered identifiers and labels before writing metadata. The checksums detect differences in dataset version, annotation content, and label derivation. The script does not create a new random split.

## Raw dataset layout

Set `DATA_ROOT` to a directory that contains the two datasets. The following layout is used by the commands below.

```text
$DATA_ROOT/
├── bdd100k/
│   ├── images/100k/train/*.jpg
│   ├── images/100k/val/*.jpg
│   └── labels/
│       ├── det_v2_train_release.json
│       └── det_v2_val_release.json
└── nuimages/
    ├── samples/CAM_*/...
    ├── v1.0-train/*.json
    └── v1.0-val/*.json
```

The BDD100K experiment used the detection annotation files `det_v2_train_release.json` and `det_v2_val_release.json`. Alternate files are acceptable only if they reproduce every reference label checksum. The required BDD100K training images number 69,863 after the original metadata filtering. The held-out pool contains all 10,000 official validation images.

The nuImages experiment uses key camera frames and the `sample`, `sample_data`, `object_ann`, `category`, `ego_pose`, and `log` tables for both official partitions. If the tables reside under `nuimages-metadata/`, set the two version arguments to `nuimages-metadata/v1.0-train` and `nuimages-metadata/v1.0-val`.

## Reconstruct metadata

Run these commands from the repository root after installing the environment described in the README. Replace `/path/to/datasets` with the dataset directory on your machine.

```bash
export DATA_ROOT=/path/to/datasets

python scripts/prepare_data.py prepare \
  --dataset bdd100k \
  --image-root "$DATA_ROOT/bdd100k" \
  --bdd-train-labels "$DATA_ROOT/bdd100k/labels/det_v2_train_release.json" \
  --bdd-val-labels "$DATA_ROOT/bdd100k/labels/det_v2_val_release.json"

python scripts/prepare_data.py prepare \
  --dataset nuimages \
  --image-root "$DATA_ROOT/nuimages" \
  --nu-train-version v1.0-train \
  --nu-val-version v1.0-val
```

Outputs are written to `data/metadata/bdd100k/` and `data/metadata/nuimages/`. Each directory contains `train.json`, `val.json`, `test_fixed.json`, `metadata_bundle.json`, `stats.json`, and `preparation_audit.json`. Image paths are absolute paths for the current machine. The generated metadata remains outside the public source release through `.gitignore`.

The script checks every required image path. Missing images or mismatched labels stop preparation. Existing outputs are preserved unless `--overwrite` is specified. After moving the datasets to another filesystem, repeat preparation with the new `--image-root` and `--overwrite`.

## Exact experiment partitions

| Dataset | Training | Historical validation | Fixed test pool | Validation used for revised training |
| --- | ---: | ---: | ---: | ---: |
| BDD100K | 62,877 | 16,986 | 10,000 | 6,986 |
| nuImages | 60,552 | 6,727 | 16,445 | 6,727 |

The historical BDD100K validation metadata includes the 10,000 fixed test images. The reconstruction preserves this historical file so that the revision's explicit split repair can be audited. Revised training removes every training or fixed test identifier from validation before model selection. The resulting 6,986 validation images are disjoint from training and testing. The historical BDD100K checkpoint must not replace the retrained checkpoint in the final evaluation.

The nuImages partitions are disjoint. Image identifiers, rather than image basenames, determine membership. A nuImages identifier includes its camera directory. Changing a directory name, omitting a camera, or sampling a new partition changes the experiment.

## Label definitions

Both datasets use image-level object presence labels. BDD100K classes are car, pedestrian, and traffic sign. nuImages classes are car, pedestrian, and traffic cone. The nuImages metadata retains the historical field name `traffic_sign_present` for the traffic cone output slot. Do not rename this field or interpret it as a traffic sign label.

BDD100K condition labels are weather, scene, and time of day. Missing or unrecognised condition labels receive a mask value of zero. nuImages condition slots encode location, illumination, and motion. Illumination uses local clock hour and motion uses ego speed thresholds of 0.5 and 8.0 m/s. These derived labels are proxies for conditions. The copied baseline preparation functions define the exact mappings and thresholds.

## Relocate existing experiment metadata

An existing copy of the original metadata can replace annotation reconstruction. It must match the released ordered identifiers, condition mappings, and label checksums.

```bash
python scripts/prepare_data.py prepare \
  --dataset bdd100k \
  --image-root "$DATA_ROOT/bdd100k" \
  --existing-metadata /path/to/original/bdd100k_metadata

python scripts/prepare_data.py prepare \
  --dataset nuimages \
  --image-root "$DATA_ROOT/nuimages" \
  --existing-metadata /path/to/original/nuimages_metadata
```

Relocation changes filesystem paths only. It preserves image identifiers, split order, condition labels, and object labels. This mode uses the Python standard library and does not import the model code.

## Reference files

`data/splits/<dataset>/` contains three ordered identifier lists and `reference.json`. The latter stores condition mappings, image counts, ordered-identifier checksums, and aggregate label checksums. It contains no per-image object or condition annotations.

The `export-reference` subcommand is a release-maintenance tool. It is not a reproduction step. Re-exporting references from a different partition would change the experiment and bypass the intended comparison.

The training and evaluation metadata serve different purposes from the 1,000-image attack manifests. Dataset preparation reconstructs the full pools. The manifest stage selects the common attack evaluation subset using the CSA model and records the selected identifiers for reuse across attacks and models.
