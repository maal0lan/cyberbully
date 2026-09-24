# Cyberbullying Detection System — Implementation Documentation

**Model:** DistilBERT-base-uncased  
**Task:** Binary Text Classification (Cyberbullying vs. Not Cyberbullying)  
**Target Performance:** 97–98% Accuracy

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Selection](#2-model-selection)
3. [Text Preprocessing](#3-text-preprocessing)
4. [Data Loading & Label Mapping](#4-data-loading--label-mapping)
5. [Class Imbalance Handling](#5-class-imbalance-handling)
6. [Data Splitting Strategy](#6-data-splitting-strategy)
7. [Tokenization & Dataset Pipeline](#7-tokenization--dataset-pipeline)
8. [Optimizer & Scheduler Configuration](#8-optimizer--scheduler-configuration)
9. [Training Loop Design](#9-training-loop-design)
10. [Threshold Tuning](#10-threshold-tuning)
11. [Early Stopping](#11-early-stopping)
12. [Evaluation Strategy](#12-evaluation-strategy)
13. [Model Persistence & Inference](#13-model-persistence--inference)
14. [Reproducibility](#14-reproducibility)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)

---

## 1. Project Overview

This system fine-tunes a pre-trained transformer model to classify social media text as either cyberbullying or non-cyberbullying. The pipeline addresses the core challenges of this problem domain: class imbalance, noisy social media text, and the need for a deployment-ready model with a well-calibrated decision threshold.

---

## 2. Model Selection

**Implementation:** `DistilBertForSequenceClassification` (distilbert-base-uncased)

**Why DistilBERT:**

DistilBERT is a distilled (compressed) version of BERT that retains approximately 97% of BERT's language understanding ability while being 40% smaller and 60% faster. For a classification task on short social media texts, the marginal performance difference between DistilBERT and full BERT does not justify the significantly higher compute cost.

The model is initialised from HuggingFace's pre-trained weights, meaning it already has deep contextual understanding of English language. Fine-tuning on the cyberbullying dataset then specialises these representations for the detection task — this transfer learning approach consistently outperforms training a model from scratch, especially on datasets of moderate size.

> **Upgrade path:** Replacing `distilbert-base-uncased` with `bert-base-uncased` in the config is expected to yield an additional 1–2% accuracy at the cost of roughly 2× training time.

---

## 3. Text Preprocessing

**Implementation:** `clean_text()` function

```python
text = demoji.replace_with_desc(text, sep=" ")  # 😡 → "enraged face"
text = re.sub(r"http\S+", "[url]", text)         # URLs → placeholder
text = re.sub(r"@\w+", "[user]", text)           # mentions → placeholder
text = re.sub(r"#(\w+)", r"\1", text)            # strip hashtag symbol
text = re.sub(r"\d+", " ", text)                 # remove numbers
```

**Why this approach:**

BERT-family models are sensitive to input quality, but they also rely on punctuation for subword tokenisation. Unlike traditional NLP pipelines, we deliberately **keep punctuation** rather than stripping it, as BERT uses it to break words into meaningful subword units (e.g., "don't" → "don", "'", "t").

Key decisions:

- **Emoji → text description** (via `demoji`): Emojis carry strong sentiment signal in cyberbullying contexts. Converting them to text descriptions (e.g., 💀 → "skull") preserves this signal rather than discarding it entirely.
- **URLs and mentions → placeholders**: These carry no semantic information relevant to classification, but their *presence* can be informative. Using `[url]` and `[user]` as placeholder tokens preserves structural information while reducing vocabulary noise.
- **Hashtags → plain text**: The `#` symbol itself is not meaningful to BERT; the word following it is. Stripping the symbol ensures the word is processed normally.
- **Numbers removed**: Numeric tokens add noise without contributing to the bullying vs. non-bullying distinction.

---

## 4. Data Loading & Label Mapping

**Implementation:** `load_data()` function with `to_binary()` mapping

**Why automatic column detection:**

Real-world cyberbullying datasets use inconsistent column naming conventions (`tweet_text`, `comment`, `message`, `cyberbullying_type`, `label`, etc.). The auto-detection logic searches for common keywords in column names, making the pipeline robust to different dataset sources without manual configuration changes.

**Why binary mapping:**

Some datasets provide multi-class labels (e.g., religion-based, gender-based, ethnicity-based bullying as separate classes). For this implementation, all bullying sub-types are collapsed into a single positive class. This decision was made because:

1. The detection objective is binary (is this harmful or not?), not taxonomic.
2. Multi-class training on heavily imbalanced sub-categories typically degrades overall performance.
3. A binary model is significantly simpler to calibrate, deploy, and explain.

---

## 5. Class Imbalance Handling

**Implementation:** `compute_class_weight` + custom `CrossEntropyLoss`

```python
w = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([w[0], w[1]], dtype=torch.float).to(DEVICE)
)
```

**Why this is the most critical fix in the pipeline:**

Cyberbullying datasets are inherently imbalanced — non-bullying content typically outnumbers bullying content by a 4:1 to 6:1 ratio. Without intervention, a model trained with standard cross-entropy loss will be implicitly rewarded for predicting the majority class, resulting in high overall accuracy but near-zero recall on the minority (bullying) class.

`compute_class_weight("balanced")` calculates inverse-frequency weights: the minority class receives a proportionally higher weight, penalising the model more severely for misclassifying bullying instances.

**The critical implementation detail:** The standard HuggingFace training loop uses `outputs.loss`, which is computed internally by the model and **silently ignores any class weights passed to it**. This pipeline correctly bypasses `outputs.loss` and instead computes loss explicitly via the custom `loss_fn(out.logits, labels)`. This single change is the primary driver of improved minority-class performance.

---

## 6. Data Splitting Strategy

**Implementation:** Stratified 80 / 10 / 10 train/val/test split

```python
X_tr, X_tmp, y_tr, y_tmp = train_test_split(..., test_size=0.20, stratify=labels)
X_val, X_te, y_val, y_te = train_test_split(..., test_size=0.50, stratify=y_tmp)
```

**Why stratified splitting:**

With an imbalanced dataset, a random split risks placing most minority-class examples in the training set, leaving the validation and test sets without adequate representation. `stratify=labels` ensures that each split preserves the original class ratio, giving reliable validation and test metrics.

**Why a dedicated test set:**

The validation set is used during training for checkpoint selection and threshold tuning — it therefore influences training decisions. Using the same data for final evaluation would produce optimistically biased metrics. A held-out test set that is never touched during training provides an unbiased estimate of real-world performance.

---

## 7. Tokenization & Dataset Pipeline

**Implementation:** `DistilBertTokenizerFast` + `BullyDataset(Dataset)`

```python
enc = self.tokenizer(
    self.texts[idx],
    truncation=True,
    padding="max_length",
    max_length=self.max_len,
    return_tensors="pt",
)
```

**Why `MAX_LEN = 128`:**

Analysis of social media text shows that the vast majority of posts fall well under 128 tokens. Setting a maximum of 128 (rather than BERT's maximum of 512) reduces memory consumption and speeds up training by approximately 4× without measurable accuracy loss on this domain.

**Why `DistilBertTokenizerFast`:**

The "Fast" tokenizer is implemented in Rust via HuggingFace Tokenizers and is substantially faster than the Python-based alternative — important when tokenising tens of thousands of samples on each epoch.

---

## 8. Optimizer & Scheduler Configuration

**Implementation:** AdamW with selective weight decay + linear warmup schedule

```python
no_decay = ["bias", "LayerNorm.weight"]
params = [
    {"params": [...], "weight_decay": WEIGHT_DECAY},  # weights
    {"params": [...], "weight_decay": 0.0},            # bias + LayerNorm
]
optimizer = AdamW(params, lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.10),
    num_training_steps=total_steps,
)
```

**Why AdamW over Adam:**

AdamW decouples weight decay from the gradient update, which is the correct formulation for transformer fine-tuning. Standard Adam applies weight decay incorrectly (conflated with the gradient), leading to suboptimal regularisation.

**Why exclude bias and LayerNorm from weight decay:**

This is the standard BERT fine-tuning recipe from the original paper. Bias terms and LayerNorm parameters are not subject to weight decay because they are scale parameters — penalising them would interfere with normalisation and destabilise training.

**Why linear warmup (10% of steps):**

Large pre-trained models are sensitive to learning rate spikes at the start of fine-tuning. A warmup period linearly increases the learning rate from near-zero to the target LR, allowing the model's weights to adjust gradually before full-speed updates begin. Without warmup, early gradient steps can push the model far from its pre-trained initialisation, destroying the representations learned during pre-training.

**Why `lr = 2e-5`:**

This is the learning rate recommended in the original BERT paper for fine-tuning on classification tasks. It is small enough to avoid catastrophic forgetting of pre-trained representations while being large enough to adapt the model meaningfully.

---

## 9. Training Loop Design

**Implementation:** Custom `train_epoch()` with gradient clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```

**Why gradient clipping at 1.0:**

Transformer models can produce large gradient magnitudes when fine-tuning, particularly in early epochs. Clipping gradients to a maximum norm of 1.0 prevents gradient explosion, which would cause unstable weight updates and potential training divergence. This is standard practice for all transformer fine-tuning.

---

## 10. Threshold Tuning

**Implementation:** Precision-Recall curve analysis on validation set

```python
prec, rec, thr = precision_recall_curve(val["labels"], val["probs"])
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_threshold = float(thr[np.argmax(f1s)])
```

**Why not use 0.5 as the default threshold:**

A threshold of 0.5 assumes equal class priors and equal costs of false positives and false negatives. Neither assumption holds here. Given class imbalance, the model's raw probability outputs are biased toward the majority class, so the optimal decision boundary is rarely at 0.5.

By sweeping the threshold across the precision-recall curve on the validation set and selecting the point that maximises F1-score, we find a threshold that best balances precision and recall for the actual data distribution. This tuned threshold is saved to `config.json` and used at inference time.

---

## 11. Early Stopping

**Implementation:** Patience-based early stopping on validation F1

```python
if val["f1"] > best_f1:
    best_f1 = val["f1"]
    patience_ctr = 0
    torch.save(model.state_dict(), ...)
else:
    patience_ctr += 1
    if patience_ctr >= PATIENCE:
        break  # stop training
```

**Why monitor F1, not accuracy:**

Given class imbalance, accuracy is a misleading metric — a model that always predicts "cyberbullying" would score over 80% accuracy on a dataset with 80% positive examples. F1-score (harmonic mean of precision and recall) penalises the model for ignoring either class and is a much more reliable indicator of true classification quality.

**Why save the best checkpoint rather than the final epoch:**

The model that minimises validation loss or maximises validation F1 mid-training often generalises better than the model at the final epoch, which may have begun overfitting to the training distribution. Saving and restoring the best checkpoint ensures the final evaluated model is the most generalised one found during training.

---

## 12. Evaluation Strategy

**Implementation:** Dual-threshold reporting on held-out test set

The final evaluation reports metrics under both the default 0.5 threshold and the tuned threshold found on the validation set. This transparency allows direct comparison and ensures the threshold tuning decision is justified by measurable improvement.

Reported metrics include Accuracy, Precision, Recall, F1-score, ROC-AUC, full Classification Report, and Confusion Matrix — providing a complete picture of model behaviour across both classes.

---

## 13. Model Persistence & Inference

**Implementation:** `model.save_pretrained()` + `config.json` + `predict()` function

```python
# config.json stores all inference-time parameters
{
  "threshold": 0.6423,
  "model": "distilbert-base-uncased",
  "class_weights": {"0": 2.47, "1": 0.62},
  "max_len": 128
}
```

**Why save threshold alongside the model:**

A saved model without its calibrated threshold is incomplete. The threshold is as much a part of the final system as the model weights. Storing it in `config.json` ensures that any future inference using `load_saved_model()` automatically uses the correct decision boundary without requiring recomputation.

---

## 14. Reproducibility

**Implementation:** Fixed seeds across all libraries

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Fixing seeds across Python's `random`, NumPy, and PyTorch (both CPU and CUDA) ensures that data splits, weight initialisations, and dropout masks are identical across runs. This makes results reproducible and debugging tractable.

---

## 15. Known Limitations & Future Work

| Limitation               | Description                                                                                                                          | Suggested Fix                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| Digit removal            | Stripping all numbers may remove age-related bullying signals ("you're 12, go back to school")                                       | Replace with number normalisation instead of removal     |
| Val F1 tracking uses 0.5 | `eval_epoch` internally uses a fixed 0.5 threshold for F1 tracking, slightly underestimating true val F1 during checkpoint selection | Pass the current best threshold into `eval_epoch`<br>    |
| No augmentation          | Class weights help imbalance but minority class is not oversampled                                                                   | Add `RandomOverSampler` from `imbalanced-learn`          |
| Binary only              | Multi-class bullying types (gender, race, religion) are collapsed                                                                    | Train a multi-class head for finer-grained detection     |
| No adversarial examples  | Model may be fooled by deliberate obfuscation (e.g., "k!ll y0u")                                                                     | Include adversarially perturbed samples in training data |

---

*Documentation generated from `cyberbully_final.py` — DistilBERT fine-tuning pipeline for binary cyberbullying detection.*
