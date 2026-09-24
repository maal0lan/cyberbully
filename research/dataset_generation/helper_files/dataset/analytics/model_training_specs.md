# Cyberbullying Model Training Specifications & Feature Guide

This document defines the complete technical specifications, feature schema, label taxonomy, preprocessing protocols, and training recommendations for training Machine Learning and Deep Learning models on the merged cyberbullying dataset.

---

## 1. Dataset Profile & Origin

- **Primary Merged File**: [`cyberbullying_merged_dataset.csv`](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/cyberbullying_merged_dataset.csv)
- **Total Samples**: **29,210** rows
- **Total Columns**: **13** columns
- **File Size**: ~5.20 MB
- **Source Breakdown**:
  - **Explicit Dataset**: 17,097 samples (58.53%) — High-intensity explicit harassment, slurs, threats, identity-based hate, and vulgar aggression.
  - **Non-Explicit Dataset**: 12,113 samples (41.47%) — Covert, subtle cyberbullying including gaslighting, backhanded compliments, rumor spreading, and social exclusion, alongside benign counter-examples.
- **Augmentation Breakdown**:
  - **Original Samples**: 28,428 (97.32%)
  - **Augmented / Adversarial Noise**: 782 (2.68%) — Synthetic adversarial perturbations (leetspeak, character deletion, character insertion, keyboard typos, spacing noise, mixed script).

---

## 2. Input Features (`X`)

### 2.1 Primary Text Feature
| Feature Name | Type | Description | Recommended Handling |
| :--- | :--- | :--- | :--- |
| `text` | `string` | The social media comment / message to classify. | Primary input for Tokenizers (BERT / DeBERTa) and Vectorizers (TF-IDF). |

#### Text Length & Token Statistics:
- **Mean Character Length**: 97.3 characters
- **Median Character Length**: 93.0 characters
- **95th Percentile Char Length**: 154.0 characters (Max: 317 characters)
- **Mean Word Count**: 17.6 words
- **Median Word Count**: 17.0 words
- **95th Percentile Word Count**: 28.0 words (Max: 58 words)
- **Tokenizer `max_length` Recommendation**: **`max_length = 128`** covers 100% of all sentences without truncation while minimizing padding overhead and GPU memory consumption.

### 2.2 Text Preprocessing & Cleaning Guidelines
When preparing `text` for Transformer models (e.g., DistilBERT, BERT, DeBERTa):
1. **Preserve Emojis with Demoji**:
   - Convert emojis to descriptive textual representations (e.g., `😡` $\rightarrow$ `enraged face`, `💀` $\rightarrow$ `skull`, `😂` $\rightarrow$ `face with tears of joy`). Emojis carry high-signal sentiment and sarcasm markers in cyberbullying contexts.
   - Do **not** blindly strip emojis.
2. **Normalize Mentions & URLs**:
   - Replace `@username` with `[user]` or `@user`.
   - Replace `http://...` / `https://...` with `[url]`.
3. **Preserve Punctuation**:
   - Transformer subword tokenizers (WordPiece, Byte-Pair Encoding) rely on punctuation (`?`, `!`, `...`) to segment clauses and detect emotional tone.
4. **Adversarial Robustness (DO NOT Over-clean Noise)**:
   - 782 samples contain synthetic leetspeak (e.g., `@$$h0le`, `k1ll`), keyboard noise, and spacing variations.
   - **Do not** apply aggressive spell-checkers or autocorrect during preprocessing, as the model must learn to detect adversarial evasion attempts in live social networks.

### 2.3 Tabular & Contextual Auxiliary Features (Optional for Hybrid Models)
If using a multi-modal or tabular-assisted architecture (e.g., Tabular-Transformer or LightGBM text-meta model):
- `char_length`: Integer length of message.
- `word_count`: Number of whitespace-separated tokens.
- `uppercase_ratio`: Ratio of capitalized characters to total length (indicates shouting/aggression).
- `punctuation_density`: Ratio of `!`, `?`, `*` to text length.
- `target_type`: Targeted entity classification (`individual`, `group`, `self`, `none`, `topic`).
- `dataset_source`: Origin of sample (`explicit` vs `non_explicit`).
- `is_augmented`: Binary flag (0 = natural text, 1 = perturbed text).

---

## 3. Cyberbullying Labels (`y`)

### 3.1 Primary Objective: Binary Classification (`gen_label`)
The primary classification task is binary detection of cyberbullying behavior.

| Value | Label Name | Count | Percentage | Class Description |
| :---: | :--- | :---: | :---: | :--- |
| **`0`** | **Not Cyberbullying** | **10,227** | **35.01%** | Benign communication, compliments, healthy constructive criticism, general venting without targeted abuse, and harmless sarcasm between friends. |
| **`1`** | **Cyberbullying** | **18,983** | **64.99%** | Targeted abuse, harassment, identity insults, threats, gaslighting, mob pile-on, social exclusion, and malicious mocking. |

#### Recommended Loss Weights for Class Imbalance:
Because cyberbullying (`1`) outnumbers non-cyberbullying (`0`) at approximately a **65:35 ratio**, models may develop a slight false-positive bias without class balancing:
- **Balanced Class Weights Formula**: $w_c = \frac{N}{K \cdot N_c}$
- **Calculated Weights**:
  - Class `0` (Not Cyberbullying): **`1.4281`**
  - Class `1` (Cyberbullying): **`0.7694`**
- **PyTorch Implementation**:
  ```python
  import torch
  class_weights = torch.tensor([1.4281, 0.7694], dtype=torch.float).to(device)
  criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
  ```

---

### 3.2 Granular Multi-Class Taxonomy (`category`)
The dataset includes **20 granular semantic categories** that capture both overt harassment and subtle interpersonal manipulation. These can be used for multi-task learning or secondary classifier heads:

| Category | Primary Label (`gen_label`) | Sample Count | Nature & Semantic Characteristics |
| :--- | :---: | :---: | :--- |
| `venting_no_target` | 0 | 3,253 | Expressing frustration about situations/life without attacking any person. |
| `topic_opinion_negative` | 0 | 3,225 | Negative opinions about media, movies, food, or sports (not individuals). |
| `sarcasm_among_friends` | 0 | 2,463 | Playful teasing between friends with positive or neutral intent. |
| `threat` | 1 | 2,452 | Explicit threats of violence, harm, or retaliation. |
| `meta_commentary_condemning_hate` | 0 | 2,448 | Anti-hate speech standing up against bullies or condemning racism/sexism. |
| `targeted_insult_identity` | 1 | 2,446 | Attacks targeting race, religion, gender, sexual orientation, disability. |
| `targeted_insult_general` | 1 | 2,434 | Direct insults attacking appearance, intelligence, or competence. |
| `backhanded_compliment` | 1 | 862 | Subtle passive-aggressive insults disguised as compliments. |
| `supportive_message` | 0 | 838 | Affirmative and uplifting statements. |
| `manipulation_gaslighting` | 1 | 835 | Psychological manipulation making the victim doubt their perception/sanity. |
| `neutral_statement` | 0 | 830 | Factual, mundane social media posts. |
| `genuine_compliment` | 0 | 826 | Sincere praise and positive feedback. |
| `social_exclusion` | 1 | 824 | Publicly ostracizing or alienating an individual from groups or activities. |
| `meta_commentary_condemning_bullying` | 0 | 820 | Comments defending victims and opposing cyberbullying behaviors. |
| `constructive_criticism` | 0 | 807 | Polite, actionable feedback without malice. |
| `pile_on_mob` | 1 | 804 | Coordinated harassment or dogpiling on a single user. |
| `implicit_mockery` | 1 | 794 | Sarcastic mockery and indirect ridicule. |
| `rumor_spreading` | 1 | 777 | Spreading unverified, damaging claims about someone's personal life. |
| `explicit_insult_clean` | 1 | 739 | Blunt insults without explicit profanity. |
| `threat_clean` | 1 | 733 | Intimidation without explicit vulgarity. |

---

### 3.3 Target Entity Taxonomy (`target_type`)
Specifies the recipient or focus of the comment:
- `individual` (18,115 rows) — Attacks or messages aimed directly at a single person.
- `none` (4,188 rows) — General venting or broadcast statements.
- `self` (3,044 rows) — Self-deprecating humor or self-reflection.
- `group` (1,695 rows) — Messages directed at communities, minorities, or teams.
- `topic` (1,144 rows) — Discussions centered on specific themes, politics, or media.

---

## 4. Critical ML Hygiene: Data Leakage Prevention

> [!CAUTION]
> **Do not use random `train_test_split()` across rows!**
>
> The dataset contains **augmented groups** marked by `augmentation_group_id`. An original text and its noisy versions (leetspeak, typos) share the exact same `augmentation_group_id`.
> If you perform standard random splitting, an augmented duplicate could land in the test set while the original is in the training set, causing **severe test set data leakage** and artificially inflated accuracy.

### Recommended Grouped Splitting Strategy:
Use `GroupShuffleSplit` or `StratifiedGroupKFold` from `sklearn.model_selection` grouped on `augmentation_group_id`:

```python
from sklearn.model_selection import GroupShuffleSplit

# Step 1: 80% Train, 20% Temp (Val + Test)
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, temp_idx = next(gss_outer.split(df, groups=df["augmentation_group_id"]))

train_df = df.iloc[train_idx]
temp_df  = df.iloc[temp_idx]

# Step 2: Split Temp into 50% Val, 50% Test (10% Val, 10% Test overall)
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss_inner.split(temp_df, groups=temp_df["augmentation_group_id"]))

val_df  = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

print(f"Train samples: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val samples:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test samples:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
```

---

## 5. Recommended Model Architectures

| Architecture | Model Family | Pros | Cons | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + LinearSVC** | Classical ML | Sub-second inference, lightweight, transparent weights. | Fails on subtle gaslighting, context, and sarcasm. | Quick baseline and latency-critical CPU environments. |
| **DistilBERT** (`distilbert-base-uncased`) | Transformer | Fast training, 40% smaller than BERT, retains 97% comprehension. | Slightly less nuance on subtle, non-explicit gaslighting. | **Recommended production default.** |
| **BERT-base** (`bert-base-uncased`) | Transformer | Stronger subword representation, high accuracy on explicit insults. | Higher memory usage (~110M params). | High accuracy requirement. |
| **DeBERTa-v3-base** (`microsoft/deberta-v3-base`) | Disentangled Attention | **State-of-the-Art** on fine-grained language nuance and subtle attacks. | Slower training; requires GPU. | **Best for nuanced/non-explicit cyberbullying detection.** |

---

## 6. Training Hyperparameter Recommendations

```python
CONFIG = {
    "model_name": "distilbert-base-uncased", # or "microsoft/deberta-v3-base"
    "max_length": 128,                       # 100% sentence coverage
    "batch_size": 16,                        # Use 8 if GPU memory < 6GB
    "epochs": 4,                             # Converges rapidly (3-5 epochs)
    "learning_rate": 2e-5,                   # AdamW learning rate
    "warmup_ratio": 0.10,                    # 10% linear warmup
    "weight_decay": 0.01,                    # L2 regularization
    "early_stopping_patience": 2,            # Stop if Val Macro F1 does not improve
    "class_weights": [1.4281, 0.7694],       # Address 65:35 class imbalance
    "random_seed": 42,
}
```

---

## 7. Model Evaluation Metrics

Do **not** evaluate solely on Accuracy, as class imbalance can conceal poor minority-class recall. Report the following metrics:

1. **Macro F1-Score**: Primary metric of model quality, giving equal weight to cyberbullying and non-cyberbullying.
2. **Per-Class Precision & Recall**:
   - **Recall on Cyberbullying (1)**: Crucial for child/user safety (minimizes missed attacks).
   - **Precision on Cyberbullying (1)**: Minimizes false flags and unnecessary censorship of benign banter.
3. **PR-AUC (Precision-Recall Area Under Curve)**: Superior to ROC-AUC under class imbalance.
4. **Adversarial Robustness Score**:
   - Calculate accuracy on the subset `is_augmented == 1` (~782 samples) to quantify resistance to leetspeak and noisy perturbations.
5. **False Positive Edge Case Breakdown**:
   - Measure false positive rates specifically on:
     - `sarcasm_among_friends`
     - `meta_commentary_condemning_hate`
     - `venting_no_target`
   - High false positives in these categories indicate over-reliance on surface-level curse words rather than true semantic intent.

---

## 8. Visual Analytics Artifacts

All visualization plots and numeric summaries are generated and maintained in the `analytics` directory:
- [01_label_distribution.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/01_label_distribution.png) — Overall class balance and source composition.
- [02_category_distribution.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/02_category_distribution.png) — Distribution of all 20 categories by label.
- [03_word_frequency_unigrams.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/03_word_frequency_unigrams.png) — Comparative top 25 unigrams.
- [04_word_frequency_bigrams.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/04_word_frequency_bigrams.png) — Comparative top 20 bigrams.
- [05_text_length_and_tokens.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/05_text_length_and_tokens.png) — Length densities, token counts, and boxplots.
- [06_augmentation_analysis.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/06_augmentation_analysis.png) — Synthetic perturbation frequencies.
- [07_target_type_distribution.png](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/plots/07_target_type_distribution.png) — Distribution of targeted entities.
- [summary_statistics.json](file:///c:/Users/Praveen/cyberbully/dataset_generation/helper_files/dataset/analytics/summary_statistics.json) — Full programmatic metrics report.
