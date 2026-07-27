# Adversarial Attack Evaluation of a nuImages MoE Router

This repository evaluates adversarial robustness of a trained
condition-specialized nuImages Mixture-of-Experts (MoE) model. Attacks target
the learned router/gating network and measure their effect on routing, individual
experts, and fused object-presence predictions.

The baseline predicts car, pedestrian, and traffic-cone presence using experts
specialized for location, illumination, and ego-motion. Since nuImages has no
ground-truth label identifying which expert the router must select, the clean
router top-1 decision is used as the reference for router-flip attacks.

## Repository contents

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── nuimages_moe_train.py
    ├── nuimages_moe_test.py
    ├── nuimages_attack_common.py
    ├── nuimages_moe_decision_attack.py
    ├── nuimages_moe_score_attack.py
    └── nuimages_moe_transfer_attack.py
```

- `nuimages_attack_common.py` contains shared dataset, model-loading, metric, and
  reporting utilities. Keep it in the same directory as every attack script.
- `nuimages_moe_decision_attack.py` supports HopSkipJump and Boundary attacks
  through Foolbox.
- `nuimages_moe_score_attack.py` supports Square and NES score-based attacks.
- `nuimages_moe_transfer_attack.py` trains a surrogate router and transfers PGD
  adversarial examples to the target router.
- The baseline train/test scripts are included because the attack code imports
  their model and evaluation helpers.

## Required baseline assets

For the initial model training, testing, and data-preprocessing implementation,
refer to the [baseline CSA-GMoE
repository](https://github.com/badjie90/CSA-GMoE-Context-Aware-Multi-Label-Object-Presence-Prediction-in-Autonomous-Driving.git).
Those baseline scripts are essential dependencies of this project: they define
the model architecture, construct the fixed metadata splits, train the original
MoE checkpoint, and provide evaluation utilities reused by the attack scripts.
Use both repositories to obtain the complete pipeline from data preparation and
baseline training through adversarial evaluation.

Run the baseline repository first. This project requires:

```text
/path/to/baseline-assets/
├── nuimages_metadata/
│   ├── metadata_bundle.json
│   └── test_fixed.json
└── nuimages_moe_stage3/
    ├── config.json
    └── checkpoints/
        └── best.pt
```

The metadata contains image paths, so the downloaded nuImages camera data must
remain accessible. Obtain it from the [official nuImages download
page](https://www.nuscenes.org/nuimages), which requires registration/login for
the full data. Do not put raw nuImages data in this repository.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended:

```bash
git clone <ATTACK-REPOSITORY-URL>
cd <ATTACK-REPOSITORY>
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
numpy>=1.26
pillow>=10.0
matplotlib>=3.8
scikit-learn>=1.4
timm>=1.0.15
torch>=2.1
torchvision>=0.16
torchmetrics>=1.3
foolbox>=3.3.4
```

Install a CUDA-compatible PyTorch build using the [official PyTorch
installer](https://pytorch.org/get-started/locally/) when necessary.

## Configure paths

Run all commands from the repository root:

```bash
METADATA_DIR=/path/to/baseline-assets/nuimages_metadata
RUN_DIR=/path/to/baseline-assets/nuimages_moe_stage3
```

Always pass these paths explicitly. The source scripts retain absolute defaults
from their original development environment.

## Recommended run order

The attacks are independent. First verify clean performance, then run attacks
with a small sample count before launching full evaluations.

### 1. Verify clean baseline performance

```bash
python scripts/nuimages_moe_test.py \
  --train-script scripts/nuimages_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/clean \
  --batch-size 64 \
  --device cuda:0
```

### 2. Decision-based attacks

HopSkipJump example:

```bash
python scripts/nuimages_moe_decision_attack.py \
  --train-script scripts/nuimages_moe_train.py \
  --test-script scripts/nuimages_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/decision_hsj \
  --attack hsj \
  --eps 0.30 \
  --steps 150 \
  --queries 2500 \
  --attack-batch-size 1 \
  --max-samples 50 \
  --device cuda:0
```

Use `--attack boundary` for Boundary Attack. HopSkipJump can consume substantial
memory; retain `--attack-batch-size 1` initially. A negative `--eps` requests an
attack run without Foolbox epsilon filtering.

### 3. Score-based attacks

Square Attack example:

```bash
python scripts/nuimages_moe_score_attack.py \
  --train-script scripts/nuimages_moe_train.py \
  --test-script scripts/nuimages_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/score_square \
  --attack square \
  --eps 0.09 \
  --queries 1000 \
  --restarts 2 \
  --max-samples 50 \
  --device cuda:0
```

Change `--attack square` to `--attack nes` for NES. Available controls include
margin or top-k-drop loss, kappa, early stopping, Square patch settings, and NES
sampling, learning-rate, sigma, momentum, and gradient normalization settings:

```bash
python scripts/nuimages_moe_score_attack.py --help
```

### 4. Transfer-based PGD attack

```bash
python scripts/nuimages_moe_transfer_attack.py \
  --train-script scripts/nuimages_moe_train.py \
  --test-script scripts/nuimages_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/transfer_pgd \
  --max-train-samples 2000 \
  --max-samples 50 \
  --sur-epochs 50 \
  --eps 0.08 \
  --pgd-steps 100 \
  --pgd-restarts 3 \
  --device cuda:0
```

This script learns a small surrogate from target-router pseudo-labels, attacks
the surrogate, and measures transfer to the original router.

## Metrics and outputs

Attack output directories contain some or all of the following:

- clean and adversarial prediction CSV files;
- router flip and attack-success rates;
- clean/adversarial router distributions and agreement;
- fused and per-expert classification metrics;
- condition-specific and object-specific results;
- clean-to-adversarial degradation summaries;
- perturbation and black-box query summaries;
- aggregate tables across multiple checkpoints or seeds.

For fair comparisons, use the same checkpoint, `test_fixed.json`, sample subset,
seed, perturbation norm and epsilon, and query budget. Do not compare query-based
methods at different budgets without reporting the difference.

## Practical guidance

- Begin with `--max-samples 10` to estimate time and memory.
- Reduce batch size when CUDA memory is insufficient.
- Pixel-space attacks operate on images in `[0, 1]` before model normalization.
- Keep each attack configuration in a separate output directory.
- Do not publish original or adversarial nuImages frames unless the dataset
  terms permit that use.
