# Controlled adversarial evaluation of routed MoE models

This repository contains the implementation for **Controlled Evaluation of Routing and Prediction Failures in Driving Mixture-of-Experts Models**. It evaluates image-level object presence on BDD100K and nuImages using CSA-GMoE and a custom sparse token-routed MoE.

The experiments compare router-targeted and fused-output-targeted Square, HopSkipJump, and transfer-PGD attacks. Four paired evaluations combine clean or adversarial expert features with clean or adversarial routing. The first letter denotes the feature state and the second denotes routing. Thus, CC is clean inference, AA is attacked inference, and AC restores clean routing on attacked features.

The sparse model is not V-MoE or Switch. The interventions separate routing from combined encoder and expert corruption. They do not isolate individual experts or establish a causal effect of condition supervision.

## Repository contents

| Directory | Contents |
| --- | --- |
| `revision/` | Attack implementations, models, metrics, training, queues, and reporting |
| `baseline/` | CSA-GMoE architecture and original metadata helpers |
| `scripts/` | Data preparation, checkpoint installation, execution, and paper plots |
| `configs/` | Protocol, recorded model configurations, and local path template |
| `data/splits/` | Ordered split identifiers and label checksums |
| `data/manifests/` | The exact 1,000 evaluation image IDs for each dataset |
| `reference_results/` | Reported aggregate tables and three selected figures |
| `artifacts/checkpoints.json` | Checkpoint identities, SHA-256 checksums, and installation paths |
| `tests/` | Metric, attack, data, and checkpoint checks |

The completed release checks are recorded in [validation notes](docs/VALIDATION.md). Run `sha256sum -c SHA256SUMS` to verify the prepared source files before changing them.

Images, annotations, weights, response banks, and raw attack outputs are excluded from Git. Selected weights are packaged separately as `checkpoints.tar.gz`. The authors must attach this file to a public release or archive before claiming that the checkpoints are publicly available. No public asset URL is embedded in this source package.

## 1. Install the environment

Use Linux, Python 3.9.25, and a CUDA-capable GPU for the full experiment. The recorded environment used PyTorch 2.8.0 with CUDA 12.8. CPU execution supports tests, figure generation, and the bounded model checks.

Run all commands from the repository root after downloading or cloning it.

```bash
conda create -n moe-eval python=3.9.25 pip -y
conda activate moe-eval
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pytest -q
```

Choose a compatible CUDA wheel if the server cannot use CUDA 12.8. Record that change when comparing results. The supported installation commands are listed in the [PyTorch version documentation](https://pytorch.org/get-started/previous-versions/).

Install `tmux` through the server package manager if needed. Reserve at least 250 GiB for generated experiment outputs, in addition to the datasets. Query snapshots retain internal tensors and account for much of this storage. Runtime depends on the hardware and attack success pattern.

## 2. Obtain the datasets and configure paths

Download the images and annotations from the [BDD100K provider](https://bdd-data.berkeley.edu/) and [nuImages provider](https://www.nuscenes.org/nuimages). Follow their access terms. The preparation code reads the annotation JSON files directly and does not require the nuImages SDK.

```bash
cp configs/paths.example.env configs/paths.env
```

Edit `configs/paths.env` to set the two dataset roots. This local file is excluded from Git. Optional output and metadata paths use repository-relative defaults. Exported environment variables take precedence over this file. Run `source scripts/common.sh` after configuration to load the paths for Python commands. The expected layout is

```text
bdd100k/
  images/100k/train/
  images/100k/val/
  labels/det_v2_train_release.json
  labels/det_v2_val_release.json
nuimages/
  samples/
  v1.0-train/
  v1.0-val/
```

The BDD100K files must be the detection annotations. nuImages requires its official training and validation images and metadata. Its official validation partition supplies the fixed evaluation pool in this experiment. See [dataset details](docs/DATA.md) for alternative directory layouts.

## 3. Reconstruct the exact metadata

```bash
source scripts/common.sh
python scripts/prepare_data.py prepare --dataset bdd100k \
  --output-dir "$BDD_METADATA" \
  --image-root "$BDD_IMAGE_ROOT" \
  --bdd-train-labels "$BDD_IMAGE_ROOT/labels/det_v2_train_release.json" \
  --bdd-val-labels "$BDD_IMAGE_ROOT/labels/det_v2_val_release.json"
python scripts/prepare_data.py prepare --dataset nuimages \
  --output-dir "$NUIMAGES_METADATA" \
  --image-root "$NUIMAGES_IMAGE_ROOT"
```

If the nuImages tables are inside `nuimages-metadata/`, add

```bash
--nu-train-version nuimages-metadata/v1.0-train \
--nu-val-version nuimages-metadata/v1.0-val
```

The preparation script checks every required image, preserves the released split order, and verifies label checksums. It stops when images, labels, or mappings differ. Outputs use the configured metadata directories, which default to `data/metadata/`, and include `preparation_audit.json`.

| Dataset | Training | Validation used for selection | Fixed test pool |
| --- | ---: | ---: | ---: |
| BDD100K | 62,877 | 6,986 | 10,000 |
| nuImages | 60,552 | 6,727 | 16,445 |

The historical BDD100K metadata has 16,986 validation rows, including all 10,000 fixed-test images. Preparation preserves that metadata for traceability. The revised training scripts remove the overlap before checkpoint selection. Use `scripts/01_train_bdd_csa.sh` for BDD100K training, not the legacy baseline training command.

## 4. Install the selected checkpoints

Obtain `checkpoints.tar.gz` from the asset location published by the repository authors. Place it outside the source tree, then run

```bash
source scripts/common.sh
python scripts/install_checkpoints.py /path/to/checkpoints.tar.gz \
  --output-dir "$OUTPUT_DIR"
```

The installer verifies archive and member checksums, then installs all four models under `$OUTPUT_DIR/models/`, which defaults to `results/models/`. It accepts identical existing files and refuses to replace different checkpoints. The released tensors match the evaluated models. Optimizer state and machine-specific embedded configuration were removed, so serialized file hashes differ from the originals. Both hashes are recorded in the checkpoint manifest.

Using the selected weights reproduces evaluation without retraining. To reproduce training instead, follow [the training procedure](docs/TRAINING.md). Retraining may produce different selected weights. Do not mix results from different checkpoints in one output directory.

## 5. Install the fixed evaluation cohorts

```bash
source scripts/common.sh
python scripts/install_manifests.py --dataset bdd100k
python scripts/install_manifests.py --dataset nuimages
bash scripts/00_preflight.sh
bash scripts/09_smoke_models.sh cpu
```

Use `bash scripts/00_preflight.sh --cpu` on a machine without CUDA. The bounded model check uses one image and four Square queries per model. It checks that CC and AA reproduce native inference and does not start the full experiment.

The installed manifests contain exactly 1,000 images each. All attacks and both models reuse these IDs and their order. The release stores identifiers only. Installation joins them to the locally reconstructed labels and image paths. Do not replace these manifests with newly sampled cohorts when reproducing the reported tables.

## 6. Run the attack queues in tmux

Start a tmux session from the repository root with the environment active.

```bash
tmux new-session -s moe-evaluation
conda activate moe-eval
source scripts/common.sh
bash scripts/08_run_all_queues.sh 0 1 2
```

For one GPU, use `bash scripts/08_run_all_queues.sh 0`. The script runs the following queues sequentially and assigns one worker to each supplied GPU. It audits each queue before advancing.

| Queue | Model | Scheduled rows |
| --- | --- | ---: |
| `primary` | CSA-GMoE | 30 |
| `controls` | CSA-GMoE | 30 |
| `transfer-scaling` | CSA-GMoE | 120 |
| `sparse-primary` | Custom sparse MoE | 30 |
| `sparse-controls` | Custom sparse MoE | 30 |
| `sensitivity` | CSA-GMoE | 240 |

These 480 scheduled rows contain 400 distinct runs. Overlapping conditions reuse completed artifacts. Direct attacks additionally produce 800 query snapshots.

Detach with **Ctrl-b**, then **d**. You can close the laptop after detaching. The server must remain running. Reconnect with

```bash
tmux attach -t moe-evaluation
```

The paths below assume the default `OUTPUT_DIR=results`. With an override, generated files use that directory instead. Progress logs are under `$OUTPUT_DIR/logs/`. An ended tmux session does not establish successful completion. The final queue audit must pass. Do not launch a second queue on GPUs already used by the first.

To run only one phase, use the detached worker launcher

```bash
bash scripts/04_generate_jobs.sh primary csa
bash scripts/05_launch_workers.sh "$OUTPUT_DIR/jobs/primary_csa.tsv" 0 1 2
```

For this route, wait for completion and audit it before launching the next phase.

```bash
bash scripts/07_audit_jobs.sh "$OUTPUT_DIR/jobs/primary_csa.tsv"
```

After an interruption, rerun the same queue command. Workers skip complete artifact families. They move partial outputs to `results/failed_artifacts/` before retrying. A changed checkpoint, cohort, or protocol requires a separate output directory.

## 7. Regenerate metrics and figures

After all queues pass, run reporting inside tmux

```bash
tmux new-session -s moe-report
conda activate moe-eval
source scripts/common.sh
bash scripts/06_report.sh
python scripts/generate_tables.py --tables-dir "$OUTPUT_DIR/tables" \
  --output-dir "$OUTPUT_DIR/paper/Tables"
python scripts/generate_figures.py --tables-dir "$OUTPUT_DIR/tables" \
  --output-dir "$OUTPUT_DIR/paper/Figures"
```

The report recomputes metrics and 5,000 paired image bootstrap replicates from `results/raw_runs/` and `results/query_budgets/`. Aggregate tables alone cannot recompute those intervals. Selected manuscript tables and figures are written under `results/paper/`.

To regenerate only the selected paper displays from the included aggregate results, run

```bash
python scripts/generate_tables.py
python scripts/generate_figures.py
```

This shorter route requires no datasets, checkpoints, GPU, or new attacks. It reproduces displays from stored statistics rather than recomputing the statistics.

## 8. Check the reproduced outputs

`results/tables/metrics_summary.csv` contains means, between-run sample SD, bootstrap bounds, and valid replicate counts. `matched_target_contrasts.csv` reports paired attack-objective contrasts. `query_budget_results.csv` and `surrogate_scaling_results.csv` report query and acquisition scaling.

For router-targeted Square at `8/255` and 1,000 queries, the reported values are

| Dataset | Route ASR | AA macro F1 | AC minus AA macro F1 |
| --- | ---: | ---: | ---: |
| BDD100K | 1.0000 | 0.7156 | 0.0116 |
| nuImages | 0.9940 | 0.5694 | 0.0472 |

These values are reference results, not universal acceptance tolerances. Check image IDs, checkpoint identity, seeds, budgets, library versions, and raw validity checks before interpreting a discrepancy. GPU kernels and hardware can change numerical outcomes.

The transfer implementation stores acquisition images in float16. The worker that creates a bank trains from the freshly acquired float32 images. Later workers load the stored float16 images. Bank creation order can therefore change surrogate training and transfer results. This release preserves the evaluated implementation.

## Protocol and interpretation

The primary pixel-space `Linf` budget is `8/255`. Sensitivity uses `2/255`, `4/255`, `8/255`, and `0.09`. Five attack or surrogate seeds are `42` through `46`. Target models use one training seed.

Direct attacks count every queried image, including clean references and initialization. Query results at 100, 250, 500, and 1,000 are prefixes of the same trajectory. HSJ may query outside the perturbation ball during boundary search, but its returned adversarial example must satisfy the bound. Transfer uses a shared 2,000-image acquisition bank and makes no target calls during subsequent PGD generation. The acquisition response contains both the route label and fused probabilities.

Bootstrap intervals resample paired image IDs while holding the five observed runs fixed. Between-run SD separately measures attack or surrogate variation. Neither quantity estimates target-model training variability. See `configs/protocol.json` for the recorded settings.

## Citation and license status

`CITATION.cff` provides software citation metadata. Add the article DOI when available. No software license has been assigned to this repository. Dataset and dependency licenses remain separate.
