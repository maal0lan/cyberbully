# Cyberbullying Detection Research Report

## Executive Summary

This report documents the current active cyberbullying detection model and summarizes the evidence saved in the repository for the final experimental run. The active artifact is `cyberbully_v0.2_run`, which is the authoritative model configuration and evaluation output used for the project context in this repository.

The project goal is a binary text-classification task: determine whether a message contains cyberbullying (label `1`) or not (label `0`). The saved model is a fine-tuned `distilbert-base-uncased` classifier trained on the generated dataset at `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`.

Based on the live saved metrics in `cyberbully_v0.2_run/metrics.json`, the model achieves strong held-out performance on the test split:

- Default threshold (`0.5`): accuracy `0.95494`, macro-F1 `0.95493`, ROC-AUC `0.98919`, PR-AUC `0.98652`
- Validation-selected threshold (`0.27`): accuracy `0.95413`, macro-F1 `0.95410`, ROC-AUC `0.98919`, PR-AUC `0.98652`

The tuned threshold increases bullying recall while slightly lowering precision, which is often desirable in safety-oriented moderation tasks where false negatives are costlier than false positives.

---

## 1. Problem Definition

The project addresses the problem of detecting cyberbullying in text-based online communication. The task is framed as a binary classification problem:

- `0` = Not cyberbullying
- `1` = Cyberbullying

The model is designed to process short and noisy social media-style text and make a risk assessment about whether the content contains bullying or harmful harassment.

---

## 2. Source Materials and Evidence Base

This report is grounded in the live repository artifacts and saved outputs rather than historical notes alone. The key evidence used in this report is:

- `cyberbully_v0.2_run/run_config.json`
- `cyberbully_v0.2_run/metrics.json`
- `cyberbully_v0.2_run/train.csv`
- `cyberbully_v0.2_run/val.csv`
- `cyberbully_v0.2_run/test.csv`
- `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`

Older notes and previous run folders are retained only as historical context and are not treated as the active final result unless explicitly identified as such.

---

## 3. Dataset Description

### 3.1 Data source

The active final run uses the expanded generated dataset at:

- `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`

This dataset contains 50,005 rows.

### 3.2 Label distribution

From the saved dataset itself:

- Total rows: `50,005`
- Positive (`gen_label = 1`): `31,726`
- Negative (`gen_label = 0`): `18,279`
- Positive share: `63.45%`
- Negative share: `36.55%`

This indicates a moderate class imbalance, with a noticeable majority of bullying-positive examples in the generated corpus.

### 3.3 Saved split sizes

The active run output contains the following split sizes:

- Train: `39,588` rows
- Validation: `4,949` rows
- Test: `4,949` rows

The validation and test sets each contain approximately 5,000 examples and are balanced enough to support robust evaluation of generalization.

---

## 4. Model Architecture and Training Setup

### 4.1 Model

The active model is:

- `distilbert-base-uncased`

This is a distilled transformer model selected to balance performance and efficiency.

### 4.2 Training configuration

The saved run configuration records the following values:

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

These settings are consistent with the repository training pipeline and the saved final run artifact.

### 4.3 Training behavior

The recorded history across the four epochs shows consistent gains in both training and validation performance:

| Epoch | Train Loss | Train Macro-F1 | Val Macro-F1 | Val Accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.70198 | 0.88009 | 0.94702 | 0.94706 |
| 2 | 0.26812 | 0.96294 | 0.95857 | 0.95858 |
| 3 | 0.15874 | 0.98330 | 0.96322 | 0.96322 |
| 4 | 0.10183 | 0.99209 | 0.96625 | 0.96626 |

This indicates that the model learned quickly and continued to improve through the final epoch without signs of major instability.

---

## 5. Preprocessing and Labeling Strategy

The dataset and run configuration indicate that the project uses social-media text normalization and binary labeling based on the generated label field. The saved run config explicitly records:

- `label_source`: `category`
- `aux_weight`: `0.3`

The project notes describe the processing pipeline as including general text normalization for noisy online content, while the final run config provides the measurable training setup. The evidence in the repository is consistent with a text-classification pipeline built for short, noisy, and sometimes adversarial or stylized social messages.

---

## 6. Decision Thresholds and Operating Points

The final run stores a validation-selected decision threshold of:

- `0.27` (stored as `0.26999999999999996` in the JSON)

The default threshold of `0.5` is also reported and evaluated for comparison.

This means the model can be used in one of two modes:

1. Default classification threshold: `0.5`
2. Validation-tuned threshold: `0.27`

The threshold selection was tuned on validation data to maximize macro-F1, then the resulting checkpoint was evaluated on the held-out test set.

---

## 7. Hold-Out Evaluation Results

The test-set metrics in `cyberbully_v0.2_run/metrics.json` are summarized below.

### 7.1 Default threshold (`0.5`)

| Metric | Value |
|---|---:|
| Accuracy | 0.95494 |
| Macro-F1 | 0.95493 |
| Bullying precision | 0.94615 |
| Bullying recall | 0.96550 |
| Non-bullying precision | 0.96424 |
| Non-bullying recall | 0.94422 |
| ROC-AUC | 0.98919 |
| PR-AUC | 0.98652 |

Confusion matrix:

- `[[2319, 137], [86, 2407]]`

Interpretation of the matrix order:

- Row 1: `[TN, FP]`
- Row 2: `[FN, TP]`

### 7.2 Validation-selected threshold (`0.27`)

| Metric | Value |
|---|---:|
| Accuracy | 0.95413 |
| Macro-F1 | 0.95410 |
| Bullying precision | 0.93745 |
| Bullying recall | 0.97393 |
| Non-bullying precision | 0.97245 |
| Non-bullying recall | 0.93404 |
| ROC-AUC | 0.98919 |
| PR-AUC | 0.98652 |

Confusion matrix:

- `[[2294, 162], [65, 2428]]`

This threshold improves recall for bullying detection at the expense of some precision. That is a common trade-off for moderation systems in which missing harmful content may be more costly than a few false positives.

---

## 8. Robustness and Adversarial Check

The live metrics file also includes an adversarial robustness section. It reports:

- Augmented rows: `113`
- Augmented accuracy: `0.9115`
- Clean accuracy: `0.9551`
- Perturbed accuracy: `0.9191`
- Clean macro-F1: `0.9551`
- Perturbed macro-F1: `0.9191`
- Prediction flip rate: `0.06245`

This suggests the model retains a meaningful amount of robustness under perturbation, though the performance drops under adversarial augmentation. The prediction flip rate is modest, indicating the model is not extremely unstable to the tested perturbations, but it is also not immune to them.

---

## 9. Interpretation of Results

The project currently demonstrates a strong binary classifier for cyberbullying detection on the repository’s generated dataset. The model performs well on the held-out test set with both default and tuned thresholds, and the threshold tuning behavior is consistent with an explicit goal of improving recall for harmful content detection.

The model’s strongest performance appears to be in the following areas:

- High ROC-AUC (`0.98919`)
- High PR-AUC (`0.98652`)
- Strong macro-F1 near `0.955`
- Stable training progression over four epochs

The most relevant trade-off is between bullying precision and bullying recall. The tuned threshold increases recall substantially while reducing precision slightly, which may be appropriate depending on deployment goals.

---

## 10. Project Scope and Evidence Boundaries

This report intentionally limits itself to the saved project artifacts and the live run evidence. It does not claim external benchmark superiority, human-annotated real-world validation, or generalization beyond the repository’s generated dataset.

Important boundaries:

- The active final run is the saved `cyberbully_v0.2_run` artifact.
- Historical notes and earlier run outputs are retained as references, not as active ground truth.
- External validation and independently collected deployment data are not asserted in the repository evidence.
- This report reflects the saved model behavior and should be interpreted as a project-specific experimental summary.

---

## 11. Reproducibility and Artifact Inventory

The repository includes the primary run outputs in the folder:

- `cyberbully_v0.2_run/`

This folder includes:

- `run_config.json`
- `metrics.json`
- `train.csv`
- `val.csv`
- `test.csv`
- `training_history.png`
- `confusion_matrix.png`
- `tokenizer/`
- `encoder_config`
- `per_category_errors.csv`
- `relabel_review.csv`

These files are the main saved artifacts for reproducing or validating the current model behavior.

---

## 12. Conclusion

The current active cyberbullying detection model is a fine-tuned DistilBERT classifier trained on a generated social-media style dataset, and the saved run artifacts show strong performance on a held-out test split. The final configuration and metrics support the claim that the model is a practical classifier for project-level research and experimentation, with both threshold choices performing strongly and the tuned threshold improving recall for harmful content detection.

The repository evidence supports the following summary statement:

> The active cyberbully detection model demonstrates strong classification performance on the repository’s generated dataset, with a macro-F1 near 0.955 and an ROC-AUC near 0.989, while preserving a useful trade-off between precision and recall through threshold tuning.

---

## 13. Appendices

### Appendix A: Verified project facts

- Final active model: `cyberbully_v0.2_run`
- Dataset rows: `50,005`
- Positive label count: `31,726`
- Negative label count: `18,279`
- Model: `distilbert-base-uncased`
- Max length: `128`
- Epochs: `4`
- Learning rate: `2e-05`
- Validation-selected threshold: `0.27`
- Default threshold: `0.5`
- Test set size: `4,949`

### Appendix B: Key file references

- `notes/AI_CONTEXT_BUNDLE.md`
- `cyberbully_v0.2_run/run_config.json`
- `cyberbully_v0.2_run/metrics.json`
- `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`

