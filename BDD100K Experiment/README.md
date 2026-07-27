# Decomposing-Adversarial-Robustness-in-Routed-Mixture-of-Experts-Models
Decomposing Adversarial Robustness in Routed Mixture-of-Experts Models: A Routing-Aware Evaluation Under Black-Box Attacks


# Adversarial Attack Evaluation of a BDD100K MoE Router

This repository evaluates the adversarial robustness of a trained
condition-specialized BDD100K Mixture-of-Experts (MoE) model. The attacks target
the router/gating network and measure how route changes affect the individual
experts and the fused MoE object-presence predictions.

Because BDD100K does not provide a ground-truth expert-routing label, the clean
router's top-1 expert is used as the reference decision for router attacks.



## Files in this repository

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── bdd100k_moe_train.py
    ├── bdd100k_moe_test.py
    ├── bdd100k_moe_decision_attack.py
    ├── bdd100k_moe_score_attack.py
    └── bdd100k_moe_transfer_attack.py
```

The training and test scripts are included because the attack programs import
the model architecture, dataset helpers, and evaluation functions from them.

- `bdd100k_moe_decision_attack.py`: HopSkipJump and Boundary attacks through
  Foolbox.
- `bdd100k_moe_score_attack.py`: query-limited Square and NES attacks.
- `bdd100k_moe_transfer_attack.py`: trains a surrogate router and transfers a
  PGD adversarial example to the target router.





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




## Prerequisites from the baseline repository

Run the baseline project first. This repository needs:

```text
/path/to/baseline-assets/
├── metadata/
│   ├── metadata_bundle.json
│   └── test_fixed.json
└── moe_stage3/
    ├── config.json
    └── checkpoints/
        └── best.pt
```

The dataset can be obtained from the [official BDD100K download
page](https://bdd-data.berkeley.edu/download.html). The attack evaluation reads
images through paths stored in the prepared metadata, so those image paths must
remain valid on the evaluation machine.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

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

Install the appropriate CUDA-enabled PyTorch build from the [official PyTorch
installer](https://pytorch.org/get-started/locally/) when necessary.

## Configure input paths

Run commands from the repository root:

```bash
METADATA_DIR=/path/to/baseline-assets/metadata
RUN_DIR=/path/to/baseline-assets/moe_stage3
```

Always pass input and output paths explicitly. The original scripts contain
absolute default paths from their development environment.

## Recommended run order

There is no mandatory order among the attacks. First verify the checkpoint with
the clean baseline evaluator, then run one or more attacks. Start with a small
sample count to estimate runtime and memory use.

### 1. Confirm clean baseline performance

```bash
python scripts/bdd100k_moe_test.py \
  --train-script scripts/bdd100k_moe_train.py \
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
python scripts/bdd100k_moe_decision_attack.py \
  --train-script scripts/bdd100k_moe_train.py \
  --test-script scripts/bdd100k_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/decision_hsj \
  --attack hsj \
  --eps 0.30 \
  --steps 150 \
  --queries 2500 \
  --max-samples 50 \
  --batch-size 16 \
  --device cuda:0
```

For Boundary Attack, change `--attack hsj` to `--attack boundary`. A negative
`--eps` requests an attack run without Foolbox epsilon filtering.

### 3. Score-based attacks

Square Attack example:

```bash
python scripts/bdd100k_moe_score_attack.py \
  --train-script scripts/bdd100k_moe_train.py \
  --test-script scripts/bdd100k_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/score_square \
  --attack square \
  --max-samples 50 \
  --device cuda:0
```

Use `--attack nes` for the NES evaluator. Inspect all query-budget, restart,
loss, and epsilon options with:

```bash
python scripts/bdd100k_moe_score_attack.py --help
```

### 4. Transfer attack

The transfer script learns a surrogate from target-router responses, generates
PGD examples against the surrogate, and evaluates their transfer to the target:

```bash
python scripts/bdd100k_moe_transfer_attack.py \
  --train-script scripts/bdd100k_moe_train.py \
  --test-script scripts/bdd100k_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/transfer_pgd \
  --eps 0.09 \
  --pgd-steps 100 \
  --pgd-restarts 3 \
  --device cuda:0
```

Run `python scripts/bdd100k_moe_transfer_attack.py --help` to select surrogate
training size, epochs, learning rate, batch size, and PGD settings supported by
the checked-out script.

## Metrics and outputs

Depending on the attack, output directories contain:

- clean and adversarial prediction CSV files;
- attack success and router flip rates;
- router top-1 agreement and route distributions;
- fused and expert-level performance under attack;
- clean-to-adversarial degradation summaries;
- perturbation and query-budget summaries;
- aggregate reports when multiple run directories are supplied.

Do not compare attacks using different test samples or budgets without stating
that difference. Preserve the attack command, seed, checkpoint, fixed test
metadata, epsilon, norm, and query budget with every reported result.

## Practical notes

- Black-box attacks can be extremely slow. Begin with `--max-samples 10`.
- Reduce batch size if CUDA memory is insufficient.
- Use `--device cpu` only for small diagnostic runs.
- The attack epsilon is applied in pixel space in scripts that operate on
  unnormalized images in `[0, 1]`.
- Do not publish BDD100K images with adversarial outputs unless dataset terms
  permit it.

