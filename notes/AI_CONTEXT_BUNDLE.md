# Cyberbullying Detection Project Context

This file reflects the current, saved project evidence. It intentionally prioritizes the live run artifacts in `cyberbully_v0.2_run/` and treats older notes as historical context only.

## 1) Project goal

Build a binary text-classification model that predicts whether a message contains cyberbullying or not.

- Label `0`: Not cyberbullying
- Label `1`: Cyberbullying

## 2) Active model and evidence source

The active final artifact is:

- `cyberbully_v0.2_run`

The model config and metrics are stored in:

- `cyberbully_v0.2_run/run_config.json`
- `cyberbully_v0.2_run/metrics.json`

The final run used the expanded generated dataset at:

- `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`

## 3) Dataset summary

From the saved dataset itself:

- Total rows: `50,005`
- Positive (`gen_label = 1`): `31,726`
- Negative (`gen_label = 0`): `18,279`
- Approximate label ratio: `63.45%` positive, `36.55%` negative

The saved train/validation/test splits in the run output are:

- Train: `39,588` rows
- Validation: `4,949` rows
- Test: `4,949` rows

## 4) Model setup

The active model is a fine-tuned `distilbert-base-uncased` classifier.

Key configuration from `run_config.json`:

- Model: `distilbert-base-uncased`
- Max token length: `128`
- Batch size: `16`
- Gradient accumulation steps: `1`
- Epochs: `4`
- Learning rate: `2e-05`
- Warmup ratio: `0.1`
- Weight decay: `0.01`
- Patience: `2`
- Auxiliary loss weight: `0.3`
- Label source: `category`
- Seed: `42`

## 5) Threshold behavior

The saved threshold in the final run config is:

- `0.27` (stored as `0.26999999999999996` in JSON)

This is the validation-selected operating point. The fixed default threshold of `0.5` is also reported in the run metrics for comparison.

## 6) Hold-out evaluation metrics

The test-set metrics in `cyberbully_v0.2_run/metrics.json` are:

### Default threshold (`0.5`)

- Accuracy: `0.95494`
- Macro-F1: `0.95493`
- Bullying precision: `0.94615`
- Bullying recall: `0.96550`
- Non-bullying precision: `0.96424`
- Non-bullying recall: `0.94422`
- ROC-AUC: `0.98919`
- PR-AUC: `0.98652`
- Confusion matrix: `[[2319, 137], [86, 2407]]`

### Validation-selected threshold (`0.27`)

- Accuracy: `0.95413`
- Macro-F1: `0.95410`
- Bullying precision: `0.93745`
- Bullying recall: `0.97393`
- Non-bullying precision: `0.97245`
- Non-bullying recall: `0.93404`
- ROC-AUC: `0.98919`
- PR-AUC: `0.98652`
- Confusion matrix: `[[2294, 162], [65, 2428]]`

The confusion-matrix ordering in the saved run is:

- Row 1: `[TN, FP]`
- Row 2: `[FN, TP]`

## 7) Training progress

The recorded validation and training history shows improvement across four epochs:

- Epoch 1: train loss `0.7020`, train macro-F1 `0.8801`, val macro-F1 `0.9470`, val acc `0.9471`
- Epoch 2: train loss `0.2681`, train macro-F1 `0.9629`, val macro-F1 `0.9586`, val acc `0.9586`
- Epoch 3: train loss `0.1587`, train macro-F1 `0.9833`, val macro-F1 `0.9632`, val acc `0.9632`
- Epoch 4: train loss `0.1018`, train macro-F1 `0.9921`, val macro-F1 `0.9663`, val acc `0.9663`

## 8) Robustness snapshot

From the adversarial evaluation section in `metrics.json`:

- Augmented rows: `113`
- Augmented accuracy: `0.9115`
- Clean accuracy: `0.9551`
- Perturbed accuracy: `0.9191`
- Clean macro-F1: `0.9551`
- Perturbed macro-F1: `0.9191`
- Prediction flip rate: `0.06245`

## 9) Important boundaries and caveats

- The active run is the current source of truth for model behavior.
- Older project notes, legacy reports, and historical `cyberbully_v0.1_run` artifacts are not treated as the current final result unless explicitly referenced as historical material.
- This document is meant to give project context, not to claim independent external validation or broader generalization beyond the included saved artifacts.
- If documentation elsewhere in the repo conflicts with the live run JSON, the live JSON should be treated as the authoritative value for the current model.

## 10) Quick interpretation

The active model performs strongly on the held-out split, with both fixed and tuned thresholds landing near `0.955` accuracy and `0.954` macro-F1. The tuned threshold improves bullying recall at the cost of some precision, which is often desirable for safety-related detection tasks.
