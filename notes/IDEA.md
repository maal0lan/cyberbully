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
