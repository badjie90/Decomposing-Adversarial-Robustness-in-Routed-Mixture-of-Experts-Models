# Selected checkpoints

The evaluation uses four selected checkpoints. Distribute `checkpoints.tar.gz` as a separate release asset or in a data archive. Do not commit the checkpoint files to the GitHub source repository. This source package does not imply that the weights have already been published.

| Model | Dataset | Training seed | Saved epoch index | Installed run directory |
| --- | --- | --- | --- | --- |
| CSA-GMoE | BDD100K | 42 | 3 | `results/models/bdd100k/csa_disjoint_seed42` |
| Custom sparse MoE | BDD100K | 42 | 18 | `results/models/bdd100k/sparse_seed42` |
| CSA-GMoE | nuImages | 42 | 3 | `results/models/nuimages/csa_seed42` |
| Custom sparse MoE | nuImages | 42 | 22 | `results/models/nuimages/sparse_seed42` |

The nuImages CSA trainer uses zero-based epoch indices, so its saved index 3 corresponds to the fourth epoch. The BDD100K CSA and both sparse trainers use one-based indices.

The BDD100K CSA checkpoint was retrained with validation and fixed evaluation images kept disjoint. The nuImages CSA checkpoint is the existing model used in the reported evaluation. The sparse models are custom architectures with four experts and top-2 token routing. They are not official V-MoE or Switch checkpoints.

## Install the release weights

The checkpoint bundle must be published separately before readers can download it. Once the authors attach `checkpoints.tar.gz` to a repository release, download that asset. From the repository root, pass the local file path to the installer.

```bash
source scripts/common.sh
python scripts/install_checkpoints.py /path/to/checkpoints.tar.gz \
  --output-dir "$OUTPUT_DIR"
```

The installer checks the bundle SHA256, file names, file sizes, and every file SHA256 against `artifacts/checkpoints.json`. It installs each `best.pt` under `checkpoints/` in the run directory and places the corresponding `config.json` alongside that directory. Existing identical files are accepted. A different existing file causes an error before installation.

Installation uses `--output-dir`, then the `OUTPUT_DIR` environment variable, then `results`. Relative output paths resolve against the checkout root or an explicit `--root`. Absolute paths support a separate experiment storage volume. The archive retains its recorded `results/models/` paths, and installation maps these to the chosen output directory's `models/` subtree. Temporary verification files remain on that output filesystem.

For example, this command installs the weights under `results_replication/models/` in the current checkout

```bash
python scripts/install_checkpoints.py /path/to/checkpoints.tar.gz \
  --output-dir results_replication
```

Use the same `OUTPUT_DIR` when installing manifests and running evaluation. The run directories listed above show the default `results` location.

To verify the bundle without installing its contents, run

```bash
python scripts/install_checkpoints.py /path/to/checkpoints.tar.gz \
  --output-dir "$OUTPUT_DIR" --verify-only
```

The installer uses only the Python standard library. It never unpickles the checkpoint. PyTorch is required when the evaluation loads the installed weights.

## Provenance and serialization

`artifacts/checkpoints.json` records the original checkpoint SHA256, release checkpoint SHA256, selected epoch, tensor count, and installed file paths. The release payload retains the model tensors. Sparse payloads also retain `model_config` and the dataset name required for construction. Each tensor was compared with its source using `torch.equal` after saving and loading the release file. All comparisons passed.

The release omits optimizer state, scheduler state, and the embedded training configuration. Thus, its file SHA256 differs from the original checkpoint SHA256 despite identical model tensors. The original checkpoints remain unchanged. The release files support inference and evaluation. They do not resume the original optimizer trajectory.

Portable training configurations are included under `configs/models/`. They retain numerical settings, class mappings, and recorded split audits. Machine-specific absolute paths were replaced with repository-relative paths. The bundle contains the same configurations at their installation locations.

## Reproduction limits

Using the selected weights isolates evaluation reproduction from training variation. Retraining can select different weights because accelerator kernels, library versions, and hardware can change numerical results. The reported five seeds describe attack or surrogate variation. They do not represent five independently trained target models.

The bundle contains no images, image metadata tables, transfer surrogates, query-response banks, or raw attack outputs. Prepare the datasets and run the documented evaluation queues separately.
