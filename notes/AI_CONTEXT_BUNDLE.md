# Cyberbullying Project: Consolidated AI Context

This file consolidates the Markdown notes and JSON evidence found under `notes/` and `notes for report/`. Source files are reproduced below under their repository-relative paths; they remain the source of truth.

## Reading guidance

- Treat plans, ideas, and pipeline explanations as project intent unless supported by the report dossier or saved artifacts.
- The report dossier states its evidence policy and claim boundaries; preserve those caveats when summarizing results.
- Conflicting metrics or implementation descriptions must be attributed to their source file and artifact rather than silently merged.
- Links inside reproduced Markdown are relative to the original source file location.
- Binary image files are listed by path below and are not embedded; inspect the referenced files when visual evidence is needed.
- Current final model status: `cyberbully_v0.2_run` is the active final run. It was trained on the expanded dataset at `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv` (50,005 rows). Earlier `cyberbully_v0.1_run` artifacts are retained only as historical reference material and are not used as the active result.

## Source index

- `notes/FINAL_REPORT.md`
- `notes/IDEA.md`
- `notes/PIPELINE_EXPLAINED.md`
- `notes for report/README.md`
- `notes for report/data_provenance.md`
- `notes for report/dataset_profile.md`
- `notes for report/evaluation_results.md`
- `notes for report/label_schema.md`
- `notes for report/legacy_model_reports.md`
- `notes for report/limitations_and_integrity.md`
- `notes for report/model_training.md`
- `notes for report/preprocessing_and_split.md`
- `notes for report/reproduction_and_artifacts.md`
- `notes for report/robustness_and_errors.md`
- `notes for report/source_register.md`
- `notes for report/study_scope.md`
- `notes for report/assets/json/dataset/summary_statistics.json`
- `notes for report/assets/json/independent_evaluation/pytorch_best_model_metrics.json`
- `notes for report/assets/json/legacy_keras/metrics.json`
- `notes for report/assets/json/legacy_keras/metrics_pure_dataset.json`
- `notes for report/assets/json/training_run/metrics.json`
- `notes for report/assets/json/training_run/run_config.json`

## Notes

---

### Source: `notes/FINAL_REPORT.md`

# 🧠 Cyberbullying Detection using DistilBERT

## Implementation Overview & Design Decisions

---

## 📌 1. Problem Statement

The goal of this project is to build a **binary text classification system** that detects whether a given text contains cyberbullying or not.

- Output:
    
    - `0` → Not Cyberbullying
    - `1` → Cyberbullying

---

## ⚙️ 2. Model Selection

### ✅ Implementation

- Used: **DistilBERT (distilbert-base-uncased)**

### 💡 Why?
- Pretrained transformer model (state-of-the-art NLP)
- Faster and lighter than BERT (≈40% smaller)    
- Maintains ~95% of BERT performance
- Suitable for real-time or limited GPU environments

---
## 🧹 3. Text Preprocessing
### ✅ Implementation
- Lowercasing text
- Replacing URLs → `[url]`
- Replacing mentions → `[user]`
- Removing hashtags symbol (`#`)
- Removing numbers (`\d+`)
- Removing extra whitespace
- Converting emojis → text (if `demoji` available)
### 💡 Why?
- Normalize noisy social media data
- Preserve semantic meaning (e.g., emojis → text)
- Remove irrelevant tokens (URLs, mentions)
- Simplify input for better learning

⚠️ Note:

- Numbers were removed, but can be retained for better context in some cases

---
## 📊 4. Dataset Handling

### ✅ Implementation
- Loaded dataset using pandas    
- Auto-detected text & label columns    
- Converted labels into binary format (0/1)    
- Removed null and empty values    

### 💡 Why?
- Ensure clean and consistent data    
- Handle different dataset formats flexibly    
- Avoid training errors due to missing data    

---
## ⚖️ 5. Class Imbalance Handling

### ✅ Implementation
- Used `compute_class_weight()` from sklearn
- Applied weights in `CrossEntropyLoss`
### 💡 Why?
- Dataset is imbalanced (~5:1 ratio)
- Prevent model from biasing toward majority class
- Improve recall for cyberbullying detection
---

## 🔀 6. Train / Validation / Test Split

### ✅ Implementation
- 80% → Train
- 10% → Validation
- 10% → Test
- Stratified splitting
### 💡 Why?
- Maintain class distribution across splits
- Proper evaluation of generalization
- Avoid overfitting

---
## 🔤 7. Tokenization

### ✅ Implementation
- Used `DistilBertTokenizerFast`
- Max length = 128
- Padding & truncation applied

### 💡 Why?
- Convert text → token IDs for model input
- Ensure fixed-length input
- Preserve important context within limit

---

## 🏗️ 8. Model Architecture

### ✅ Implementation

- `DistilBertForSequenceClassification`
- Output layer → 2 classes
### 💡 Why?

- Pretrained language understanding
- Fine-tuned for classification task
- Efficient and accurate

---

## 🏋️ 9. Training Strategy

### ✅ Implementation
- Optimizer: `AdamW`
- Learning rate: `2e-5`
- Scheduler: Linear warmup
- Gradient clipping
- Batch size: 16
- Epochs: 5

### 💡 Why?

- AdamW prevents overfitting via weight decay
- Warmup stabilizes early training
- Gradient clipping prevents exploding gradients
- Small LR ensures stable fine-tuning

---

## 📈 10. Evaluation Metrics

### ✅ Implementation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### 💡 Why?

- Accuracy alone is misleading (due to imbalance)
- F1-score balances precision & recall
- ROC-AUC evaluates overall performance
- Confusion matrix shows error types

---

## 🎯 11. Threshold Optimization

### ✅ Implementation

- Default threshold = 0.5
- Tuned threshold using Precision-Recall curve
- Selected threshold maximizing F1-score (~0.16)    

### 💡 Why?

- Improve balance between precision and recall
- Reduce false negatives (critical for safety task)
- Customize model behavior based on use-case

---

## 🛑 12. Early Stopping

### ✅ Implementation

- Stops training if validation F1 doesn’t improve for 3 epochs

### 💡 Why?
- Prevent overfitting
- Save training time
- Ensure best model is retained

---

## 💾 13. Model Saving

### ✅ Implementation

- Saved:
    - model weights (`.pt`)
    - tokenizer files
    - config.json (threshold, params)
### 💡 Why?

- Enable reuse without retraining
- Support deployment & inference
- Store optimal threshold

---

## 🔮 14. Inference Pipeline

### ✅ Implementation

- Clean input text
- Tokenize
- Get probability
- Apply threshold
- Output label + confidence

### 💡 Why?
- End-to-end prediction system
- Consistent with training pipeline
- Real-world usability

---

## 🚀 15. Overall Approach

### 🧠 Type:

- **Deep Learning (Transformer-based NLP)**    

### 🔥 Pipeline:

1. Data Cleaning    
2. Tokenization
3. Transformer Encoding (DistilBERT)    
4. Classification Head    
5. Threshold Optimization    

---

## 🏁 Conclusion

This project uses a **modern deep learning approach (transformers)** to detect cyberbullying with high accuracy. Key improvements such as **class weighting, threshold tuning, and early stopping** significantly enhance performance and reliability.

---

## ⭐ Key Strengths

- High performance (PR-AUC ~0.99)    
- Handles imbalance effectively    
- Optimized for real-world use    
- Lightweight yet powerful model    

---

## ⚠️ Possible Improvements

- Retain numbers instead of removing    
- Use BERT for higher accuracy    
- Hyperparameter tuning    
- Data augmentation

---

---

### Source: `notes/IDEA.md`

---
tags: [nlp, dataset, cyberbullying, llm, ollama, project, summary]
updated: 2026-09-20
status: active
related: ["[[cyberbully-dataset-pipeline]]", "[[clean-pipeline-addendum]]"]
---

# Hermes — Project Idea & Progress Summary

## The core idea

Build a better cyberbullying-detection training dataset than what's publicly available
(e.g. the ~49k-row Kaggle set), because that dataset has **label noise from keyword
shortcuts** — it correlates surface words with the bullying label instead of actual
target + intent. Symptom: "I hate football" gets flagged as bullying (no target, it's
an opinion) while genuinely targeted-but-polite-sounding bullying gets missed.

**The fix isn't more volume — it's data that breaks the word-to-label shortcut.**
That means deliberately generating *contrast pairs*: the same trigger word or topic
appearing in both a bullying and a non-bullying sentence, so a model trained on it has
to learn context/target/intent rather than vocabulary.

The project has two parallel tracks:

1. **Profanity-anchored track** — bullying/non-bullying sentences built around ~250
   explicit bad words/slurs (targeted insults, threats, vs. topic opinions, venting,
   meta-commentary, banter).
2. **Clean (no-profanity) track** — bullying/non-bullying sentences with **zero**
   profanity, since real-world bullying is mostly exclusion, mockery, rumor-spreading,
   backhanded compliments, and manipulation, not slurs. A classifier trained only on
   profanity-adjacent examples would both miss most real bullying and over-flag any
   sentence with a curse word regardless of context — same shortcut problem, inverted.

## Pipeline architecture (shared by both tracks)

```
seed list (words or topics)
      │
      ▼
1. GENERATE  — dolphin-mistral (uncensored, local via Ollama) writes sentences per
               (seed, category) pair, self-assigns a label, outputs structured JSON
      │
      ▼
2. JUDGE     — a second, different, stronger model (qwen2.5:7b-instruct) independently
               re-labels each sentence blind (no category/label shown to it)
      │
      ▼
3. FILTER    — keep only rows where generator label == judge label ("agree").
               Disagreements go to needs_review.jsonl for manual review, not discarded.
      │
      ▼
4. DEDUP     — embed sentences (all-MiniLM-L6-v2), drop near-duplicates
               (cosine similarity > 0.92) within each label class
      │
      ▼
5. BALANCE   — cap rows per (seed, category) so no single word/topic dominates
      │
      ▼
   final_dataset.csv  +  needs_review.jsonl
```

Two independent models for generate vs. judge is the key design choice — trusting the
generator's self-label alone would just reproduce the keyword-shortcut problem in a new
form. Different model families reduce correlated blind spots.

## Track 1: Profanity-anchored (done, pipeline built)

- **Seed file**: `words.txt` — ~250 explicit bad words/slurs
- **7 categories**: targeted_insult_general, targeted_insult_identity, threat,
  topic_opinion_negative (hard negative), venting_no_target (hard negative),
  meta_commentary_condemning_hate (hard negative), sarcasm_among_friends (ambiguous)
- **Files**: `generate_pipeline.py`, `templates.py`, `words.txt`
- **Status**: pipeline built and runnable, generates → judges → filters → dedups →
  balances → exports `final_dataset.csv`

## Track 2: Clean / no-profanity (in progress, checkpoint tested)

- **Seed file**: `topics.txt` — 88 traits/situations (weight, grades, accent, family,
  gaming rank, etc.) instead of bad words
- **14 categories** (expanded from 7 to cover real bullying shapes): adds
  social_exclusion, rumor_spreading, backhanded_compliment, manipulation_gaslighting,
  pile_on_mob (bullying side); genuine_compliment, constructive_criticism,
  supportive_message, neutral_statement, meta_commentary_condemning_bullying
  (non-bullying side)
- **Extra hard constraint**: post-generation **blocklist filter** — the 403 unique
  profanity words previously used in Track 1 (auto-extracted from Track 1's
  `generated_raw.csv`) are used to reject any Track 2 row that leaks a bad word, with a
  second safety check right before final export
- **Efficiency upgrades over Track 1's script**: concurrent generation/judging via
  `ThreadPoolExecutor(max_workers=6)` + `OLLAMA_NUM_PARALLEL=4` (tuned for RTX 5060 /
  16GB RAM), checkpoints every 5 topics with 3 randomly printed sample rows for live
  quality checking
- **Target**: 10,000+ final rows. Estimated ~7,500-9,000 on first full pass at current
  88-topic list; documented how to close the gap (add topics, raise `N_PER_CALL` /
  `MAX_PER_TOPIC_CATEGORY`)
- **Files**: `pipeline_clean.py`, `templates_clean.py`, `topics.txt`, `blocklist.txt`

### Checkpoint QA (first 4 topics: weight, height, acne, braces — 559 rows)

- ✅ 0 blocklist leaks
- ✅ Roughly balanced labels (313 bullying / 246 not) and even category spread
- ✅ Near-zero exact duplicates (dedup stage will catch near-dupes too)
- ⚠️ Found 1 mislabeled row in `meta_commentary_condemning_bullying` (generator
  self-labeled an anti-bullying sentence as bullying) — expected to be caught by the
  judge-disagreement filter, but flags the category as generator-sloppy
- ⚠️ Found placeholder-text leakage — `"(person's name)"` literally appearing in
  `genuine_compliment` outputs instead of a real filled-in name. Not caught by
  blocklist or dedup since it's not profanity and not a duplicate.
- **Action item, not yet applied**: add a placeholder-text filter (reject rows
  containing `"(person's name)"`, `"[name]"`, `"{name}"` etc.) before running the full
  88-topic batch

## Design principles established along the way

1. **Contrast pairs beat raw volume** — the same word/topic in both classes is what
   actually teaches a model the label boundary.
2. **Never trust a single model's self-labeling** — independent judge model is
   non-negotiable.
3. **Judge model must be equal-or-stronger than the generator, never weaker** — judging
   context/intent is harder than generating text; a weak judge (e.g. <1B params)
   reintroduces keyword-shortcut errors.
4. **Disagreements are signal, not noise** — route to manual review instead of silently
   keeping or discarding them.
5. **Balance by category/seed, not just by final label** — otherwise the dataset fills
   up with near-identical phrasing for whichever word/topic the model liked generating.
6. **Checkpoint and sample constantly** — catching a mislabeling pattern or a
   placeholder-leak bug at 559 rows is cheap; catching it at 12,000 rows is not.

## Open items / next steps

- [ ] Add placeholder-text filter to `pipeline_clean.py` before full run
- [ ] Run full 88-topic Track 2 batch, monitor checkpoints
- [ ] Manual review pass on both tracks' `needs_review.jsonl` files
- [ ] Decide: merge Track 1 + Track 2 into one combined dataset, or keep separate and
      combine at train time
- [ ] If short of 10k after Track 2's first pass: expand `topics.txt` and/or raise
      per-call and per-cap limits, re-run incrementally
- [ ] Eventually: train/fine-tune the actual classifier on the combined dataset and
      evaluate against the original noisy Kaggle-based baseline to confirm the
      shortcut-learning problem is actually fixed

---

### Source: `notes/PIPELINE_EXPLAINED.md`

# Cyberbullying Detection Pipeline — What's Actually Happening

## Why DistilBERT fine-tuning, not TF-IDF + SVC

TF-IDF + SVC (or logistic regression, or any bag-of-words classifier) scores text by
*which words appear*, weighted by how rare/common they are across the corpus. That's
the wrong tool for this dataset on purpose, because your category taxonomy is built
around cases where the **words are nearly identical but the label is opposite**:

| Text | Category | Label |
|---|---|---|
| "You're an idiot and everyone here knows it." | targeted_insult_general | 1 (bullying) |
| "Calling people idiots online is hateful, we should be better." | meta_commentary_condemning_hate | 0 (not bullying) |
| "Ugh, I hate Mondays so much." | venting_no_target | 0 |
| "You're so worthless, nobody would even notice if you left." | targeted_insult_general | 1 |

A TF-IDF vector barely distinguishes these — "idiot," "hate," "worthless" fire the same
features either way. A bag-of-words model would learn "hate = bullying" as a shortcut and
get sarcasm, meta-commentary, and venting wrong every time — which is exactly the
failure mode your `sarcasm_among_friends` / `meta_commentary_*` / `venting_no_target`
categories exist to catch (see the `FP_WATCH` list in the script).

DistilBERT reads the *sentence as a whole* — word order, negation, who's being addressed,
sarcasm markers — instead of an unordered word bag. That's the only way to separate
"I hate this movie" from "I hate you." TF-IDF+SVC would be faster to train and fine for a
first baseline, but it structurally can't solve the context-vs-keyword problem your
dataset was purpose-built around. So: DistilBERT fine-tuning it is.

---

## The model architecture

```
raw text
   │
   ▼
tokenizer (DistilBERT wordpiece)
   │
   ▼
DistilBERT encoder (pretrained, 6 layers, 66M params)  ──►  last_hidden_state
   │
   ▼
mean-pool over tokens (using the attention mask, so padding doesn't count)
   │
   ├──► binary head (Linear: 768 → 2)   →  bullying / not-bullying   [MAIN TASK]
   │
   └──► category head (Linear: 768 → 20) → which of your 20 semantic categories [AUX TASK]
```

This is **multi-task learning**: one shared encoder, two output heads. The category head
isn't the real goal — it's there to (a) force the encoder to learn *why* something is
bullying, not just *whether*, and (b) give you the `likely_category` output at inference
time so you (or a downstream LLM) have something to explain the prediction with, without
resorting to keyword matching.

The two losses are combined as:

```
total_loss = binary_cross_entropy + aux_weight * category_cross_entropy
```

`aux_weight` (default 0.3) controls how much the category task is allowed to influence
the shared encoder. Too high and the model spends capacity on a 20-way problem instead of
the binary one you actually care about; too low (0) and you lose the auxiliary signal
and the `likely_category` output becomes meaningless. This is one of the first things
worth tuning.

---

## Step-by-step: what `train_cyberbully.py` actually does

### 1. Data loading & cleaning (`load_and_prepare`)
- Reads one or more CSVs (`--data-paths`) and concatenates them.
- Strips URLs/@mentions into `[url]`/`[user]` placeholders, converts emoji to text
  (via `demoji`), collapses hashtags to plain words. **Does not lowercase or strip
  digits/symbols** — that would destroy leetspeak like `sna7ch`, `@$$h0le`, which the
  model needs to see in order to learn to be robust to it.
- Drops rows with leftover generator placeholders (e.g. `[derogatory term ...]`).
- **Re-derives the binary label from `category`**, not from `gen_label` directly, using
  the `CATEGORY_TO_LABEL` mapping in the script. This matters because ~5% of rows in the
  merged CSV have a `gen_label` that contradicts what their `category` should map to
  (mostly `meta_commentary_*` and `topic_opinion_negative` rows). Every row that gets
  relabeled is written to `relabel_review.csv` so you can spot-check it. Use
  `--label-source raw` if you'd rather trust `gen_label` as-is.
- Builds a `group` key = normalized text of the **original** sentence (before any
  augmentation). This is what keeps a sentence and all its leetspeak/typo variants
  together in the same split later.
- Drops duplicate clean sentences (same normalized text seen twice).

### 2. Train / val / test split (`grouped_split`)
- 80/10/10, using `StratifiedGroupKFold` twice (once for train vs. rest, once to split
  the rest into val/test).
- **Grouped** by the `group` key from step 1 — so `"snatch"` and its augmented copy
  `"sna7ch"` always land in the *same* split. If they didn't, the model could
  memorize the original in train and "predict" the leetspeak version in test, giving
  you a fake accuracy boost that means nothing on unseen adversarial text.
- **Stratified** by label, so all three splits keep roughly the same bullying/not-bullying
  ratio as the full dataset.
- Asserts zero group overlap between splits before continuing — if that assertion ever
  fails, something is wrong with the grouping and the script stops rather than silently
  producing a leaky result.

### 3. Tokenizing & class weighting
- Text is tokenized with the DistilBERT tokenizer, truncated to `--max-len` (128
  tokens default — your word-count chart shows most rows are well under 30 words, so
  64 would also work and train faster if you want to try it).
- Class weights for the loss are computed **from the train split only**, as
  `N / (2 * N_c)` per class — this is what stops the model from just predicting the
  majority class to get a cheap accuracy number on your ~55/45 imbalance.

### 4. Training loop
- Standard fine-tuning loop: forward pass → combined loss → backward → optimizer step →
  linear LR schedule with warmup → gradient clipping.
- Mixed precision (`bf16` on modern GPUs, `fp16` with a gradient scaler otherwise, off on
  CPU) for speed.
- `--grad-accum-steps` lets you simulate a larger batch size on a small GPU by
  accumulating gradients over N mini-batches before each optimizer step.
- After every epoch: evaluate on validation, track macro-F1 (the primary metric — better
  than accuracy for an imbalanced binary problem because it doesn't let the majority
  class dominate the score).
- Whenever val macro-F1 improves, save `best_model.pt` and re-tune the decision threshold
  on validation (searching 0.05–0.95, picking whatever maximizes macro-F1 — not
  assuming 0.5 is optimal).
- Early stopping after `--patience` epochs with no improvement.
- `last_checkpoint.pt` is saved every epoch regardless, so `--resume` can pick back up
  after a crash or manual stop without losing progress.

### 5. Final evaluation (on the held-out test split, touched only once)
- Reports accuracy, macro-F1, precision/recall for each class, ROC-AUC, PR-AUC, and the
  confusion matrix — at both the default 0.5 threshold and the one tuned on val.
- **Per-category error breakdown**: for each of your 20 categories, what fraction of
  test rows were misclassified, and whether that's a false-positive or false-negative
  (depends on the category's expected label). This tells you *which kind* of bullying
  (or non-bullying) the model struggles with, not just an aggregate score.
- **False-positive watch-list**: specifically prints the error rate for
  `sarcasm_among_friends`, `meta_commentary_condemning_hate/bullying`,
  `venting_no_target`, `topic_opinion_negative` — the categories where a model that
  learned "keyword = bullying" would fail loudest. Low error here is your real signal
  that the model learned context, not a word list.
- **Adversarial robustness**: (a) accuracy specifically on the pre-made `is_augmented`
  rows in the test set (leetspeak/typo/spacing-noise versions from your augmented CSV),
  and (b) accuracy on *freshly generated* perturbations of clean test rows — comparing
  clean vs. perturbed accuracy and reporting the prediction "flip rate."
- **Contrast probes**: ten hand-written same-topic-opposite-intent sentence pairs
  (e.g. "I hate football, boring sport" vs "you're worthless, nobody would notice")
  checked directly, printed pass/fail — a fast eyeball sanity check beyond aggregate
  numbers.

### 6. What gets saved to `--out-dir`
```
best_model.pt              best checkpoint (by val macro-F1)
last_checkpoint.pt         full resumable state (model+optimizer+scheduler+history)
tokenizer/                 tokenizer files
encoder_config/             DistilBERT config (for reloading the architecture)
run_config.json             all hyperparameters + chosen threshold + category list
metrics.json                 test metrics (default + tuned threshold), adversarial results, history
confusion_matrix.png         test confusion matrix
training_history.png         loss curve + train/val macro-F1 per epoch
train.csv / val.csv / test.csv    the actual split data, with the `group` column removed
relabel_review.csv           rows where category-derived label != raw gen_label
per_category_errors.csv       full per-category error table
test_predictions.csv          every test row with its predicted probability/label
```

### 7. Inference (`--predict`)
Loads `best_model.pt` + tokenizer + config from `--out-dir`, runs the same cleaning as
training, and returns for each input text:
```
prob_bully        confidence (0-1) from the binary head
label              "cyberbullying" / "not_cyberbullying" at the tuned threshold
likely_category    argmax of the 20-way category head
```
No keyword/lexicon matching anywhere — deliberately. The category output is meant to
stay context-level, so it doesn't fail exactly where the whole point of the dataset was
to force the model past surface words. If you want to generate a human-readable
explanation downstream, hand `(text, likely_category, prob_bully)` to another LLM and
ask it to explain why that category applies — it has real signal to work with.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. sanity check the whole pipeline runs end-to-end (1 epoch, 1500 rows, ~1 min)
python train_cyberbully.py --data-paths cyberbullying_merged_dataset.csv --smoke

# 2. just look at the cleaned/split data without training anything
python train_cyberbully.py --data-paths cyberbullying_merged_dataset.csv --prep-only

# 3. full training run
python train_cyberbully.py --data-paths cyberbullying_merged_dataset.csv \
    --out-dir ./run1 --epochs 4 --batch-size 16 --lr 2e-5

# 3b. equivalently, from separate raw + augmented files (auto-merged + deduped)
python train_cyberbully.py --data-paths cyberbully_raw_dataset.csv cyberbully_augmented_dataset.csv \
    --out-dir ./run1

# 4. resume if it got interrupted
python train_cyberbully.py --data-paths cyberbullying_merged_dataset.csv --out-dir ./run1 --resume

# 5. try it on new text
python train_cyberbully.py --out-dir ./run1 --predict "you are pathetic" "I hate this movie"
```

## What to look at after a run, in order

1. `metrics.json` → `test_tuned_threshold.macro_f1` — your headline number.
2. `per_category_errors.csv` — sorted worst-first. If the top of the list is dominated
   by `sarcasm_among_friends` / `meta_commentary_*` / `venting_no_target`, the model is
   still leaning on keywords rather than context — worth more epochs, a higher
   `aux_weight`, or examining those specific rows.
3. `adversarial` block in `metrics.json` — how much accuracy drops from clean to
   perturbed text. A big drop means the model isn't robust to the kind of obfuscation
   your augmented dataset was designed to simulate.
4. The contrast-probe pass rate printed at the end of the run — quick gut check.

## Report dossier

---

### Source: `notes for report/README.md`

# Cyberbullying Research Dossier

This folder organizes the repository evidence for a research paper and the broader project goal: build a reusable Python package for cyberbullying detection that users can install with `pip`. The project motivation is that existing packages did not fit the team's intended needs; this is a project rationale, not a systematic comparison of available packages. The current repository is still script-based and does not yet contain standard package-install metadata. The active final experimental artifact is `cyberbully_v0.2_run`, trained on the expanded generated dataset at `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv` (50,005 rows). Earlier `cyberbully_v0.1_run` outputs remain historical reference material and are not the active final model.

## Topic map

- [Study scope and research questions](study_scope.md)
- [Dataset profile](dataset_profile.md)
- [Labels and taxonomy](label_schema.md)
- [Dataset provenance](data_provenance.md)
- [Preprocessing and split design](preprocessing_and_split.md)
- [Model and training setup](model_training.md)
- [Held-out evaluation](evaluation_results.md)
- [Robustness results](robustness_and_errors.md)
- [Legacy model reports](legacy_model_reports.md)
- [Limitations and claim boundaries](limitations_and_integrity.md)
- [Reproduction and artifacts](reproduction_and_artifacts.md)
- [Source register and copied evidence](source_register.md)

## At-a-glance result

The current final DistilBERT run is `cyberbully_v0.2_run`, trained on the expanded generated dataset (`50,005` rows total). Its saved validation-selected threshold is `0.27`, and the held-out test split contains `4,949` rows. The saved report records accuracy `0.9541`, macro-F1 `0.9541`, ROC-AUC `0.9892`, and PR-AUC `0.9865` at that tuned threshold. At the fixed `0.5` threshold, the report records accuracy `0.9549` and macro-F1 `0.9549`. The two operating points are reported separately; see [evaluation](evaluation_results.md) for counts and caveats.

## Evidence policy

Numbers and implementation details here are taken from saved artifacts or executable code. Planning documents are treated as plans, not proof that every planned step was completed. The copied JSON and image files are in [assets](source_register.md). External scholarly citations, human annotation evidence, and evaluation on independently collected data were not found in the inspected repository materials and are not asserted here.

---

### Source: `notes for report/data_provenance.md`

# Dataset Provenance

## What is evidenced in the repository

The merged-dataset utility reads explicit and non-explicit CSVs, adds a `dataset_source` field, concatenates their rows, and writes a merged dataset. The analytics JSON reports 17,097 explicit and 12,113 non-explicit rows in its snapshot. The dataset also records augmentation fields such as `is_augmented`, `augmentation_type`, and `original_text`; the snapshot summary reports 782 augmented rows (2.68%).

![Source and label distribution](assets/images/dataset/01_label_distribution.png)

![Augmentation analysis](assets/images/dataset/06_augmentation_analysis.png)

## What is not established

The generator documentation describes a generate, judge, filter, deduplicate, and balance workflow, including local language models. Those documents also include open tasks and proposed settings. The available merge code establishes concatenation and provenance tagging, but does not establish that every merged row was independently judged, manually reviewed, or generated by a specified model. The merged CSV does not provide enough evidence to make those claims on its own.

Treat the generator materials as design and progress notes: [explicit-track notes](../dataset_generation/helper_files/gen/explicit%20generator/cyberbully-dataset-pipeline.md) and [clean-track addendum](../dataset_generation/helper_files/gen/non-explicit-generator/clean-pipeline-addendum.md). The merge implementation is [`merge_datasets.py`](../dataset_generation/helper_files/expander/merge_datasets.py).

## Augmentation

The analytics summary lists leetspeak, insertions, deletions, spacing noise, keyboard noise, repetition, mixed-script noise, and combinations. Because augmented examples are related to original sentences, the training code groups an augmented row with its `original_text` before splitting. The measured robustness results are summarized in [robustness](robustness_and_errors.md).

---

### Source: `notes for report/dataset_profile.md`

# Dataset Profile

## Source snapshot

The current final dataset is the generated corpus at `dataset_generation/helper_files/dataset/generated_dataset/cyberbullying_dataset.csv`, which contains 50,005 rows across 20 categories. It records 31,726 positive (`gen_label = 1`) and 18,279 negative (`gen_label = 0`) rows, or 63.45% and 36.55%, respectively. The source mix is 17,097 explicit and 12,113 non-explicit rows in the underlying source snapshot; the final training pipeline retains the merged generated dataset as its active input.

The final run uses the category-aware binary label mapping implemented in `cyberbully_final.py`; the saved training/test CSVs are derived from the generated corpus rather than from the earlier raw merged snapshot.

## Model-ready split snapshot

The active final run, `cyberbully_v0.2_run`, contains 39,588 training rows, 4,949 validation rows, and 4,949 test rows. The total model-ready dataset is 49,486 rows after cleaning and split preparation; the full raw generated source retains 50,005 rows and the remaining difference is accounted for by the pipeline's filtering and deduplication steps.

The current saved validation and test CSVs each contain 2,457 class-0 and 2,492 class-1 rows, and training contains 19,649 class-0 and 19,939 class-1 rows. These split counts reflect the final category-derived `label` column used by the active run.

## Category profile

The active dataset includes the full 20-category taxonomy used by the model: hard negatives such as `venting_no_target`, `topic_opinion_negative`, and `meta_commentary_condemning_hate`, and positive categories such as `targeted_insult_identity`, `threat`, `social_exclusion`, and `rumor_spreading`. See the current run configuration in `cyberbully_v0.2_run/run_config.json` and the final label mapping in the training code.

The current final model uses a binary detection task with an auxiliary 20-way category head, so the resolved labels are explicitly category-derived rather than trusting the raw generator labels alone.

---

### Source: `notes for report/evaluation_results.md`

# Held-Out Evaluation

## Primary saved report

The active final run is `cyberbully_v0.2_run`. It selects the decision threshold on validation data by maximizing validation macro-F1, then evaluates the checkpoint on the held-out `test.csv`. The saved validation-selected threshold is `0.27`.

| Test operating point | Threshold | Accuracy | Macro-F1 | Bullying precision | Bullying recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation-selected | 0.27 | 0.9541 | 0.9541 | 0.9375 | 0.9739 | 0.9892 | 0.9865 |
| Fixed threshold | 0.50 | 0.9549 | 0.9549 | 0.9461 | 0.9655 | 0.9892 | 0.9865 |

The validation-selected operating point has confusion matrix `[[2294, 162], [65, 2428]]`, ordered as `[[TN, FP], [FN, TP]]`. The fixed `0.50` point has `[[2319, 137], [86, 2407]]`. The saved test split contains 4,949 rows. These values come from the live run's `cyberbully_v0.2_run/metrics.json` and should be read with the training script's current threshold-selection rules.

## Separate evaluator output

The repository also includes an independent evaluator script, but the active final run is the saved `cyberbully_v0.2_run` artifact. Any threshold selected using test labels is exploratory and must not be treated as the final headline result. The threshold-free ROC-AUC and PR-AUC from the saved checkpoint remain valid summary metrics for the current run, while threshold tuning is a separate operating-point decision.

---

### Source: `notes for report/label_schema.md`

# Labels and Taxonomy

## Binary target

The training code maps the 20 semantic categories to two labels. Label `0` means not cyberbullying; label `1` means cyberbullying. With the saved run configuration's `label_source: category`, the training pipeline derives labels from this mapping. If a category is unknown to the mapping, the code retains its raw `gen_label` value and logs the unknown category.

| Label 0: not cyberbullying | Label 1: cyberbullying |
|---|---|
| `venting_no_target` | `threat`, `threat_clean` |
| `topic_opinion_negative` | `targeted_insult_identity`, `targeted_insult_general` |
| `sarcasm_among_friends` | `explicit_insult_clean`, `backhanded_compliment` |
| `meta_commentary_condemning_hate` | `manipulation_gaslighting`, `social_exclusion` |
| `meta_commentary_condemning_bullying` | `pile_on_mob`, `implicit_mockery` |
| `supportive_message`, `neutral_statement` | `rumor_spreading` |
| `genuine_compliment`, `constructive_criticism` | |

The mapping is implemented in [`cyberbully_final.py`](../cyberbully_final.py). The dataset's `category` values encode a project-defined taxonomy; the repository does not include an independent annotation handbook or inter-annotator agreement study.

## Relabeling and review

The pipeline can write rows whose raw label differs from the category-derived label to `relabel_review.csv`. The existence of this review file records candidates for inspection, not that all rows were manually reviewed or adjudicated. Do not describe category-derived labels as human-validated unless additional evidence is added.

## Scope caveat

The labels represent the project taxonomy, not a universal definition of cyberbullying. For discussion of ambiguous categories and context limits, see [limitations](limitations_and_integrity.md).

---

### Source: `notes for report/legacy_model_reports.md`

# Legacy Model Reports

The repository contains separate Keras artifacts and JSON metrics under `cyberbully_output`. These are not the DistilBERT run and should not be blended into its results.

| Copied report | Stored description | Reported accuracy | Reported F1 | Provenance caveat |
|---|---|---:|---:|---|
| [metrics.json](assets/json/legacy_keras/metrics.json) | Model type is not stated in this JSON | 0.8691 | 0.9262 | The JSON does not identify the exact checkpoint, split, or metric protocol. |
| [metrics_pure_dataset.json](assets/json/legacy_keras/metrics_pure_dataset.json) | `Keras_BiLSTM` | 0.8544 | 0.9167 | This JSON identifies a model type but does not include enough split provenance for a controlled comparison. |

These reports are preserved for auditability, not presented as a valid benchmark against the grouped-split PyTorch model. Their thresholds differ, and the repository materials inspected here do not establish matched data, split, or evaluation procedures. No claim that one architecture is superior should be based on these values alone.

---

### Source: `notes for report/limitations_and_integrity.md`

# Limitations and Claim Boundaries

## Dataset and labels

- The dataset is project-specific and includes synthetic/generated and augmented examples. The repository does not establish that its distribution represents naturally occurring platform content.
- Category-derived binary labels encode the project's taxonomy. Evidence of complete human adjudication or inter-annotator agreement was not found.
- The project notes discuss judging and manual review, but open tasks remain in those notes. Do not present those tasks as completed quality controls without supporting records.
- English `distilbert-base-uncased` inputs and this dataset do not support claims about other languages.

## Evaluation

- The test data is held out from model training within the supplied dataset, but it is not an external-domain evaluation.
- The current held-out test split has 4,949 rows from one split seed. No confidence intervals, repeated-seed results, or independent replication are included in the cited run artifacts.
- The training metrics use a threshold selected on validation data. Any threshold selected on test labels must be treated as exploratory and should not replace the saved validation-selected operating point.
- Synthetic robustness results cover only the transformations implemented by this repository and include the current augmented test subset recorded in the saved metrics JSON.
- Aggregate scores do not establish safety, fairness, calibrated risk, or suitability for automated moderation.

## Claims to avoid without new evidence

Do not claim state-of-the-art performance, real-world effectiveness, production readiness, comprehensive adversarial robustness, fairness across demographic groups, human-verified labels, causal benefit from an individual training technique, or superiority to the archived Keras models.

## Citation status

The inspected project documents do not provide a scholarly bibliography for the model, dataset, or evaluation methods. Add and verify primary academic references before submission; no external citations have been invented in this dossier.

---

### Source: `notes for report/model_training.md`

# Model and Training Setup

## Architecture

The saved run uses the pretrained `distilbert-base-uncased` encoder, attention-mask-weighted mean pooling, dropout (`0.1`), a two-class binary head, and a 20-class category head. The objective combines weighted binary cross-entropy with `0.3` times category cross-entropy. The category head is auxiliary to the binary detection task.

![Training history](assets/images/model/training_history.png)

## Saved run configuration

| Setting | Saved value |
|---|---:|
| Maximum token length | 128 |
| Batch size | 16 |
| Epochs completed | 4 |
| Learning rate | `2e-5` |
| Warmup ratio | 0.10 |
| Weight decay | 0.01 |
| Early-stopping patience | 2 |
| Auxiliary category loss weight | 0.30 |
| Random seed | 42 |
| Label source | Category mapping |
| Validation-selected threshold | 0.27 |

Binary class weights were computed from training data and saved as approximately `[1.0074, 0.9927]` for labels 0 and 1 in the current active run. The code uses AdamW, gradient clipping at norm 1.0, and a linear warmup/decay schedule. See the copied [run configuration](assets/json/training_run/run_config.json) and [training metrics](assets/json/training_run/metrics.json).

## What this does not show

Only this configuration's saved result is documented here. The repository does not provide a controlled comparison establishing that DistilBERT, the auxiliary head, class weighting, or these hyperparameters outperform alternatives. Model-selection rationale in older notes is not experimental evidence. No model was retrained as part of preparing this dossier.

---

### Source: `notes for report/preprocessing_and_split.md`

# Preprocessing and Split Design

## Text preparation

The training script converts emoji to text descriptions when `demoji` is available, replaces URLs and mentions with placeholders, removes the `#` from hashtags, and collapses repeated whitespace. It preserves case, punctuation, digits, and symbols. It filters generator placeholder text, drops duplicate non-augmented sentences based on a normalized grouping key, and stores the cleaned model input in `model_text`.

The normalization used for grouping lowercases text and removes non-alphanumeric characters before comparison. For augmented rows, grouping uses `original_text` when available; otherwise it falls back to the row's text. This distinction is intended to keep variants of one base sentence together.

## Split method

The script uses `StratifiedGroupKFold` with seed `42`: one five-fold split to form training versus the remaining data, followed by a two-fold split of the remainder for validation and test. The saved run config records the seed, and the code asserts that group identifiers do not overlap between splits.

| Split | Rows | Label 0 | Label 1 | Unique groups |
|---|---:|---:|---:|---:|
| Train | 39,588 | 19,649 | 19,939 | not reported in the saved summary |
| Validation | 4,949 | 2,457 | 2,492 | not reported in the saved summary |
| Test | 4,949 | 2,456 | 2,493 | not reported in the saved summary |

The saved split files were checked: group overlap was zero for train/validation, train/test, and validation/test. See the actual [`train.csv`](../cyberbully_v0.2_run/train.csv), [`val.csv`](../cyberbully_v0.2_run/val.csv), and [`test.csv`](../cyberbully_v0.2_run/test.csv) artifacts.

## Interpretation

The held-out test set is a split of the current generated project dataset, not an independently collected external test set. Grouping reduces one form of leakage between related original and augmented rows; it does not establish independence from source, topic, template, or generator characteristics.

---

### Source: `notes for report/reproduction_and_artifacts.md`

# Reproduction and Artifacts

## Training entry point

The training script is [`cyberbully_final.py`](../cyberbully_final.py). Its saved run configuration records the dataset path, model identifier, seed, maximum length, batch size, epochs, learning rate, warmup, weight decay, patience, auxiliary-loss weight, and label source. The active final checkpoint and tokenizer are retained in `cyberbully_v0.2_run`.

From the repository root, the documented default training command is:

```powershell
python cyberbully_final.py
```

This command can retrain and overwrite artifacts in the default run directory. Do not rerun it merely to inspect this dossier.

## Evaluation entry point

The standalone evaluator is [`eval_results/evaluate_models.py`](../eval_results/evaluate_models.py). It reloads the saved PyTorch checkpoint and defaults to the saved test CSV. Its test-threshold optimization caveat is documented in [evaluation](evaluation_results.md).

## Package status

The intended deliverable is a reusable Python package installable with `pip`. At the time these notes were prepared, the repository had not yet added standard packaging metadata or a package layout, so no `pip install` workflow is documented as working. The current scripts and `requirements.txt` are not, by themselves, evidence that this project is an installable distribution.

## Evidence copies

Copies of source JSON reports and figures are stored under [`assets`](source_register.md). They were copied without changing their contents; the source files remain in their original locations. The model weight files were not duplicated because the request concerned report evidence, and the checkpoint is already present in the repository.

---

### Source: `notes for report/robustness_and_errors.md`

# Robustness and Error Analysis

## Saved robustness measurements

The training run reports accuracy `0.9487` on 78 augmented test rows. It also reports clean accuracy `0.9720`, accuracy `0.9394` on programmatically perturbed clean test text, clean macro-F1 `0.9719`, perturbed macro-F1 `0.9392`, and prediction flip rate `0.0510`.

The training code constructs perturbations using leetspeak substitutions, character deletion/insertion, neighboring-key substitutions, and inserted spacing. These are results for the implemented synthetic transformations only; they do not establish robustness to all evasion strategies or naturally occurring language variation.

![Augmentation profile in the source dataset](assets/images/dataset/06_augmentation_analysis.png)

## Interpretation limits

The 78-row augmented subset is small. The saved report does not provide confidence intervals or significance tests for these robustness values. The perturbation routine is synthetic and shares assumptions with the training pipeline. Avoid describing the model as "robust" without this qualification.

No per-category error table is summarized here because the research claim should be based on inspecting the corresponding [`per_category_errors.csv`](../cyberbully_v0.2_run/per_category_errors.csv), including its category-level denominators. Treat that artifact as exploratory until those counts and label definitions are reviewed.

---

### Source: `notes for report/source_register.md`

# Source Register and Evidence Copies

## Copied JSON

| Dossier copy | Original source |
|---|---|
| [Dataset summary](assets/json/dataset/summary_statistics.json) | `dataset_generation/helper_files/dataset/analytics/summary_statistics.json` |
| [Training metrics](assets/json/training_run/metrics.json) | `cyberbully_v0.2_run/metrics.json` |
| [Training configuration](assets/json/training_run/run_config.json) | `cyberbully_v0.2_run/run_config.json` |
| [Standalone evaluation metrics](assets/json/independent_evaluation/pytorch_best_model_metrics.json) | `eval_results/pytorch_best_model_metrics.json` |
| [Legacy metrics](assets/json/legacy_keras/metrics.json) | `cyberbully_output/metrics.json` |
| [Pure-dataset Keras metrics](assets/json/legacy_keras/metrics_pure_dataset.json) | `cyberbully_output/metrics_pure_dataset.json` |

## Copied figures

Dataset figures: [label distribution](assets/images/dataset/01_label_distribution.png), [category distribution](assets/images/dataset/02_category_distribution.png), [unigrams](assets/images/dataset/03_word_frequency_unigrams.png), [bigrams](assets/images/dataset/04_word_frequency_bigrams.png), [text lengths](assets/images/dataset/05_text_length_and_tokens.png), [augmentation](assets/images/dataset/06_augmentation_analysis.png), and [target types](assets/images/dataset/07_target_type_distribution.png).

Model figures: [training history](assets/images/model/training_history.png), [training-run confusion matrix](assets/images/model/training_run_confusion_matrix.png), [ROC curve](assets/images/model/pytorch_best_model_roc_curve.png), [precision-recall curve](assets/images/model/pytorch_best_model_pr_auc_curve.png), [threshold sweep](assets/images/model/pytorch_best_model_accuracy_vs_threshold.png), [default-threshold confusion matrix](assets/images/model/pytorch_best_model_confusion_matrix_default.png), [test-selected-threshold confusion matrix](assets/images/model/pytorch_best_model_confusion_matrix_best_macro_f1.png), and [calibration curve](assets/images/model/pytorch_best_model_calibration_curve.png).

## Primary implementation and project sources

- [`cyberbully_final.py`](../cyberbully_final.py): cleaning, category mapping, grouping, split, architecture, training, threshold selection, and test reporting.
- [`evaluate_models.py`](../eval_results/evaluate_models.py): separate test evaluation and plot generation.
- [`model_training_specs.md`](../dataset_generation/helper_files/dataset/analytics/model_training_specs.md) and [summary statistics](../dataset_generation/helper_files/dataset/analytics/summary_statistics.json): dataset description and analytics; the prose specification includes recommendations that are not necessarily measured results.
- [`merge_datasets.py`](../dataset_generation/helper_files/expander/merge_datasets.py): explicit/non-explicit dataset merge implementation.
- [README](../README.md), [implementation notes](../cyberbullying_implementation.md), and the older [pipeline overview](../notes/PIPELINE_EXPLAINED.md): retained for context, but contain stale paths, planned work, or claims not independently supported by the current saved run. Prefer the code and copied run artifacts when they differ.

Copied files are byte-for-byte copies of the source artifacts. JSON is retained as evidence, including legacy outputs whose model/split provenance is incomplete; see [legacy reports](legacy_model_reports.md).

---

### Source: `notes for report/study_scope.md`

# Study Scope

## System under study

The documented experiment fine-tunes `distilbert-base-uncased` for binary text classification. It also trains a 20-category auxiliary prediction head. The binary output is the primary task; the category output is a second supervised task.

## Project motivation: a pip-installable Python package

The broader engineering goal is to turn the detection work into a reusable Python package that users can install with `pip`, rather than requiring them to assemble scripts and model files themselves. The team's motivation is that existing packages did not meet the needs this project is trying to address. The repository does not include a systematic survey or benchmark of those packages, so this should be presented as project motivation, not as a general finding about the Python ecosystem.

The current repository has training and prediction scripts, but no standard package metadata (`pyproject.toml`, `setup.py`, or `setup.cfg`) was found at the project root. Package publication and pip installation are therefore goals, not capabilities demonstrated by the current artifacts. See [reproduction and artifacts](reproduction_and_artifacts.md).

## Candidate research questions

These are questions the available experiment can inform, not conclusions already established:

1. How does this model classify examples from the project's merged explicit and non-explicit text dataset under a grouped, stratified holdout split?
2. How do the fixed `0.5` decision threshold and the threshold selected on validation data differ on the held-out test split?
3. How does performance change on the included augmented examples and on programmatically perturbed test text?

The repository does not contain controlled ablations for the auxiliary category head, class weighting, or grouped versus random splitting. It therefore does not establish the causal contribution of any one design choice.

## Intended paper scope

Describe a project-specific dataset and one saved model run. Do not generalize the measured scores to all social-media users, platforms, languages, or real-world moderation settings. See [limitations](limitations_and_integrity.md) and [dataset provenance](data_provenance.md).

## JSON evidence

---

### Source: `notes for report/assets/json/dataset/summary_statistics.json`

```json
{
  "dataset_overview": {
    "total_rows": 29210,
    "total_columns": 15,
    "column_names": [
      "text",
      "gen_label",
      "category",
      "target_type",
      "source_word",
      "original_source_word",
      "augmentation_type",
      "augmentation_level",
      "original_text",
      "augmentation_id",
      "augmentation_group_id",
      "is_augmented",
      "dataset_source",
      "char_len",
      "word_cnt"
    ],
    "memory_usage_mb": 19.95
  },
  "class_balance_gen_label": {
    "cyberbullying_1": 18983,
    "not_cyberbullying_0": 10227,
    "cyberbullying_pct": 64.99,
    "not_cyberbullying_pct": 35.01,
    "recommended_class_weights": {
      "0": 1.4281,
      "1": 0.7694
    }
  },
  "source_distribution": {
    "explicit": 17097,
    "non_explicit": 12113
  },
  "augmentation_summary": {
    "original_rows": 28428,
    "augmented_rows": 782,
    "augmentation_percentage": 2.68,
    "augmentation_types": {
      "leetspeak": 151,
      "insertion": 144,
      "deletion": 117,
      "spacing_noise": 86,
      "keyboard_noise": 81,
      "repetition": 74,
      "mixed_script": 52,
      "leetspeak+insertion": 9,
      "leetspeak+deletion": 8,
      "deletion+leetspeak": 6
    }
  },
  "text_statistics": {
    "char_length": {
      "mean": 97.3,
      "median": 93.0,
      "min": 12,
      "max": 317,
      "p95": 154.0
    },
    "word_count": {
      "mean": 17.6,
      "median": 17.0,
      "min": 3,
      "max": 58,
      "p95": 28.0
    }
  },
  "category_counts": {
    "venting_no_target": 3253,
    "topic_opinion_negative": 3225,
    "sarcasm_among_friends": 2463,
    "threat": 2452,
    "meta_commentary_condemning_hate": 2448,
    "targeted_insult_identity": 2446,
    "targeted_insult_general": 2434,
    "backhanded_compliment": 862,
    "supportive_message": 838,
    "manipulation_gaslighting": 835,
    "neutral_statement": 830,
    "genuine_compliment": 826,
    "social_exclusion": 824,
    "meta_commentary_condemning_bullying": 820,
    "constructive_criticism": 807,
    "pile_on_mob": 804,
    "implicit_mockery": 794,
    "rumor_spreading": 777,
    "explicit_insult_clean": 739,
    "threat_clean": 733
  },
  "target_type_counts": {
    "individual": 18115,
    "none": 4188,
    "self": 3044,
    "group": 1695,
    "topic": 1144,
    "object": 408,
    "activity": 145,
    "media": 144,
    "concept": 78,
    "general": 47
  }
}
```

---

### Source: `notes for report/assets/json/independent_evaluation/pytorch_best_model_metrics.json`

This archived evaluator snapshot is a historical reference from the earlier training run and is not the active headline artifact. The active model is the saved `cyberbully_v0.2_run` checkpoint, whose evaluation summary is recorded in `cyberbully_v0.2_run/metrics.json` and whose held-out split contains 4,949 rows.

The current final run reports default-threshold accuracy `0.9549`, macro-F1 `0.9549`, ROC-AUC `0.9892`, and PR-AUC `0.9865`, with a validation-selected threshold of `0.27` and a test confusion matrix `[[2319, 137], [86, 2407]]`.

---

### Source: `notes for report/assets/json/legacy_keras/metrics.json`

```json
{
  "threshold": 0.1623639464378357,
  "accuracy": 0.8691253951527924,
  "precision": 0.8748876909254267,
  "recall": 0.9838343015913109,
  "f1_score": 0.9261681131851147,
  "roc_auc": 0.9085484678514572,
  "confusion_matrix": [
    [
      229,
      557
    ],
    [
      64,
      3895
    ]
  ],
  "class_weights": {
    "0": 3.016912512716175,
    "1": 0.5993280452685292
  }
}
```

---

### Source: `notes for report/assets/json/legacy_keras/metrics_pure_dataset.json`

```json
{
  "model_type": "Keras_BiLSTM",
  "threshold": 0.04482509195804596,
  "accuracy": 0.8544142614601019,
  "precision": 0.8760733348804827,
  "recall": 0.9612936083524318,
  "f1_score": 0.9167071393880525,
  "roc_auc": 0.8922785095508963,
  "confusion_matrix": [
    [
      251,
      534
    ],
    [
      152,
      3775
    ]
  ],
  "test_samples": 4712,
  "training_time_sec": 1697.12
}
```

---

### Source: `notes for report/assets/json/training_run/metrics.json`

```json
{
  "test_default_threshold": {
    "threshold": 0.5,
    "accuracy": 0.9549403919983835,
    "macro_f1": 0.9549261406754387,
    "precision_bully": 0.9461477987421384,
    "recall_bully": 0.9655034095467309,
    "precision_not_bully": 0.9642411642411642,
    "recall_not_bully": 0.9442182410423453,
    "roc_auc": 0.9891938470061448,
    "pr_auc": 0.9865192290352448,
    "confusion_matrix": [
      [
        2319,
        137
      ],
      [
        86,
        2407
      ]
    ]
  },
  "test_tuned_threshold": {
    "threshold": 0.26999999999999996,
    "accuracy": 0.9541321479086684,
    "macro_f1": 0.9540984966278367,
    "precision_bully": 0.9374517374517375,
    "recall_bully": 0.9739269955876454,
    "precision_not_bully": 0.9724459516744384,
    "recall_not_bully": 0.9340390879478827,
    "roc_auc": 0.9891938470061448,
    "pr_auc": 0.9865192290352448,
    "confusion_matrix": [
      [
        2294,
        162
      ],
      [
        65,
        2428
      ]
    ]
  },
  "adversarial": {
    "augmented_rows_n": 113,
    "augmented_rows_accuracy": 0.911504424778761,
    "clean_accuracy": 0.9551282051282052,
    "perturbed_accuracy": 0.9191480562448304,
    "clean_macro_f1": 0.9550878287396177,
    "perturbed_macro_f1": 0.9190868564774148,
    "prediction_flip_rate": 0.062448304383788254
  },
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.7019823745449986,
      "train_macro_f1": 0.8800867107447101,
      "val_macro_f1": 0.9470157753188195,
      "val_acc": 0.9470600121236613,
      "epoch_seconds": 137.6826412677765
    },
    {
      "epoch": 2,
      "train_loss": 0.2681244621335557,
      "train_macro_f1": 0.9629398110438828,
      "val_macro_f1": 0.9585745068655678,
      "val_acc": 0.9585774904021014,
      "epoch_seconds": 114.66210269927979
    },
    {
      "epoch": 3,
      "train_loss": 0.1587364758065704,
      "train_macro_f1": 0.9833019368902034,
      "val_macro_f1": 0.9632236311314499,
      "val_acc": 0.9632248939179632,
      "epoch_seconds": 111.60901379585266
    },
    {
      "epoch": 4,
      "train_loss": 0.10182815375803697,
      "train_macro_f1": 0.9920930242218983,
      "val_macro_f1": 0.9662542165233615,
      "val_acc": 0.9662558092543948,
      "epoch_seconds": 112.78117775917053
    }
  ]
}
```

---

### Source: `notes for report/assets/json/training_run/run_config.json`

```json
{
  "model": "distilbert-base-uncased",
  "max_len": 128,
  "threshold": 0.26999999999999996,
  "categories": [
    "backhanded_compliment",
    "constructive_criticism",
    "cyberstalking_monitoring",
    "explicit_insult_clean",
    "fake_joke_deniability",
    "friendly_banter",
    "genuine_compliment",
    "implicit_mockery",
    "manipulation_gaslighting",
    "meta_commentary_condemning_bullying",
    "meta_commentary_condemning_hate",
    "neutral_statement",
    "pile_on_mob",
    "rumor_spreading",
    "sarcasm_among_friends",
    "sarcasm_harmless",
    "social_exclusion",
    "supportive_message",
    "targeted_insult_general",
    "targeted_insult_identity",
    "threat",
    "threat_clean",
    "topic_opinion_negative",
    "venting_no_target",
    "victim_reporting"
  ],
  "class_weights": [
    1.0073795104076544,
    0.9927278198505441
  ],
  "aux_weight": 0.3,
  "label_source": "category",
  "args": {
    "data_paths": [
      "dataset_generation\\helper_files\\dataset\\generated_dataset\\cyberbullying_dataset.csv"
    ],
    "out_dir": "./cyberbully_v0.2_run",
    "model": "distilbert-base-uncased",
    "max_len": 128,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "epochs": 4,
    "lr": 2e-05,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "patience": 2,
    "aux_weight": 0.3,
    "label_source": "category",
    "num_workers": 0,
    "seed": 42,
    "prep_only": false,
    "smoke": false,
    "resume": false,
    "predict": null
  }
}
```

## Image assets

- `notes for report/assets/images/dataset/01_label_distribution.png`
- `notes for report/assets/images/dataset/02_category_distribution.png`
- `notes for report/assets/images/dataset/03_word_frequency_unigrams.png`
- `notes for report/assets/images/dataset/04_word_frequency_bigrams.png`
- `notes for report/assets/images/dataset/05_text_length_and_tokens.png`
- `notes for report/assets/images/dataset/06_augmentation_analysis.png`
- `notes for report/assets/images/dataset/07_target_type_distribution.png`
- `notes for report/assets/images/model/pytorch_best_model_accuracy_vs_threshold.png`
- `notes for report/assets/images/model/pytorch_best_model_calibration_curve.png`
- `notes for report/assets/images/model/pytorch_best_model_confusion_matrix_best_macro_f1.png`
- `notes for report/assets/images/model/pytorch_best_model_confusion_matrix_default.png`
- `notes for report/assets/images/model/pytorch_best_model_pr_auc_curve.png`
- `notes for report/assets/images/model/pytorch_best_model_roc_curve.png`
- `notes for report/assets/images/model/training_history.png`
- `notes for report/assets/images/model/training_run_confusion_matrix.png`