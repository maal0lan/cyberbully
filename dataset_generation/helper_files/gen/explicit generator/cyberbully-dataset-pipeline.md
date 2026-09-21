---
tags: [nlp, dataset, cyberbullying, llm, ollama, project]
created: 2026-09-18
status: active
---

# Cyberbullying Dataset Generation Pipeline

## The core problem

Original Kaggle dataset (~49k rows) has class imbalance and **label noise from keyword shortcuts** — the labeling process (or a weak model used to build it) correlates surface words ("hate", "kill", slurs) with the bullying label, regardless of context or target.

Symptoms:
- `"i hate foot ball"` → labeled cyberbullying (wrong — no target, no person)
- `"i hate pedos"` → might get labeled non-bullying (wrong direction, but same root issue: word-level shortcut instead of semantic understanding)

**Fix isn't more data volume. It's data that breaks the word→label shortcut.** Need pairs where the same trigger word appears in both bullying and non-bullying contexts, with the label determined by target + intent, not vocabulary.

## Core labeling principle

Cyberbullying = **targeted** + **intent to harm/degrade** + (usually) **directed at a person/identifiable group**.

Not cyberbullying:
- Opinions about things/topics ("I hate football", "this movie sucks")
- Venting without a target ("I hate my life today")
- Discussing/condemning hate speech (meta-commentary, quoting to criticize)
- Criticism of ideas, public figures' policies/actions (not identity-based attack)
- Banter/jokes with clear mutual context (harder — often needs conversation-level context, flag as ambiguous)

Cyberbullying:
- Direct insults at a person ("you're worthless", "kys")
- Identity-based attacks (race, gender, sexuality, religion, disability)
- Threats, humiliation, exclusion tactics, doxxing threats
- Pile-on/mob language directed at an individual

## Pipeline stages

### 1. Concept templates (not word lists)
Convert your ~250 trigger words into **concept categories**, each producing multiple example *types*:
- targeted_insult_general
- targeted_insult_identity
- threat
- topic_opinion_negative (hard negative)
- venting_no_target (hard negative)
- meta_commentary_condemning_hate (hard negative)
- sarcasm_among_friends (ambiguous — hand-review bucket)

Each of your 250 words gets slotted into 1+ categories and used inside templates for multiple categories where relevant (this is what generates the *contrast pairs* that actually teach the boundary).

### 2. Generation (local, via Ollama)
- Model: `dolphin-mistral` (uncensored — needed since real bullying language is often filtered by aligned models, and we need faithful representations of both classes)
- Structured prompt → structured JSON output: `{text, label, category, target_type}`
- Batch by concept category, not by raw word, ~5-10 completions per (word, category) pair

### 3. Judging / filtering (the quality gate)
- **Never trust the generator's self-assigned label.**
- Second pass with a *different model family*, 7B+ instruct model (e.g. `qwen2.5:7b-instruct` or `llama3.1:8b-instruct`) — NOT a <1B model, judging is harder than generating and needs a model that can reason about target/intent.
- Keep rows where generator label == judge label.
- Route disagreements to a `needs_review.jsonl` file for manual pass (expect ~10-20% of rows here — that's normal and valuable, don't discard).

### 4. Dedup
- Embed with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`)
- Cosine similarity threshold ~0.92 → drop near-duplicates within same label class (LLMs love repeating structure per word)

### 5. Balance
- Balance by **category**, not just by final label — otherwise you'll just get 5000 near-identical "you are an idiot" variants for the bullying class
- Target roughly equal bullying / non-bullying, and reasonable spread across categories within each

### 6. Manual spot-check
- Stratified random sample (~100-150 rows), especially from hard-negative categories
- This 30-60 min pass catches systemic judge errors before you burn compute at scale

## Model choices — reasoning

| Role | Model | Why |
|---|---|---|
| Generator | dolphin-mistral (Ollama) | uncensored, needed for authentic toxic-class text |
| Judge | qwen2.5:7b-instruct or llama3.1:8b-instruct | different family than generator (reduces correlated errors), strong enough to reason about context — NOT a <1B model |
| Dedup embeddings | all-MiniLM-L6-v2 | fast, local, good enough for near-dup detection |
| Testing/plumbing only | any tiny model (0.5B–1B) | fine for testing JSON parsing / pipeline code speed, NOT for real labels |

## Open questions / next steps
- [ ] Decide final category taxonomy (7 above is a starting point)
- [ ] Map 250 words → categories (some words apply to multiple)
- [ ] Decide target dataset size (suggest starting smaller — 5-10k high quality — before scaling to 49k+)
- [ ] Set up Ollama with dolphin-mistral + qwen2.5:7b-instruct
- [ ] Run pipeline script (see `generate_pipeline.py`)
- [ ] Manual review pass on `needs_review.jsonl`
- [ ] Merge, balance, export final CSV

## Files
- `generate_pipeline.py` — full generation + judge + dedup + balance pipeline
- `words.txt` — your word list, one per line (add your 250 words here)
- `templates.py` — concept category → prompt templates
