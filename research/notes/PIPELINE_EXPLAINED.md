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
