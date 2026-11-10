# Reproduce target-model training

The selected checkpoint bundle is sufficient for evaluation. Training is an alternative workflow. It requires the prepared datasets, CUDA, and access to the ImageNet-pretrained ConvNeXt-V2 weights fetched by `timm`.

Use a separate output directory so the selected release checkpoints and existing attacks remain intact. Run each command from the repository root in the configured environment. Load the local paths first with `source scripts/common.sh`. The inline output overrides below take precedence over `configs/paths.env`.

```bash
OUTPUT_DIR=results_retrained bash scripts/01_train_bdd_csa.sh 0
OUTPUT_DIR=results_retrained bash scripts/01_train_nuimages_csa.sh 0
OUTPUT_DIR=results_retrained bash scripts/02_train_sparse.sh bdd100k 0
OUTPUT_DIR=results_retrained bash scripts/02_train_sparse.sh nuimages 0
```

These commands run sequentially on GPU 0. Execute them inside tmux for a long run. They refuse nonempty model directories. To use several GPUs concurrently, assign a different GPU ID to each active training process.

| Model | Epoch limit | Batch | Learning rate | Selection |
| --- | ---: | ---: | --- | --- |
| BDD100K CSA | 100 | 32 | Encoder `2e-5`, other groups `1e-4` | Validation macro AUROC |
| nuImages CSA | 50 | 32 | Encoder `2e-5`, other groups `1e-4` | Validation macro AUROC |
| Custom sparse | 50 | 64 | `3e-4` | Validation macro AUROC |

Training uses seed 42. The BDD100K trainer removes all training and test IDs from the historical validation set. The resulting validation set contains 6,986 images. The nuImages script checks that all three splits are disjoint before training. Sparse training applies the same BDD100K validation repair.

The CSA model uses condition losses on expert heads and object losses on expert and fused heads. The router does not receive a direct condition-label classification loss. Sparse training uses object BCE and load balancing without condition labels. Saved numerical configurations are under `configs/models/`.

Retraining is not expected to reproduce checkpoint bytes across hardware. The fixed evaluation IDs still define the comparison cohort. Install them into the separate output tree before evaluation

```bash
OUTPUT_DIR=results_retrained python scripts/install_manifests.py --dataset bdd100k
OUTPUT_DIR=results_retrained python scripts/install_manifests.py --dataset nuimages
OUTPUT_DIR=results_retrained bash scripts/08_run_all_queues.sh 0
OUTPUT_DIR=results_retrained bash scripts/06_report.sh
```

To study a new cohort selected by a retrained CSA model, use `OUTPUT_DIR=results_new_cohort bash scripts/03_make_manifests.sh 0 0` after placing the corresponding checkpoints in that output tree. This creates a new experiment. It does not reproduce the published evaluation cohort.
