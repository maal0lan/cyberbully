---
tags: [nlp, dataset, cyberbullying, llm, ollama, project]
created: 2026-09-20
status: active
parent: "[[cyberbully-dataset-pipeline]]"
---

# Clean (no-profanity) dataset — addendum

Companion to the original bad-word pipeline. Same generate → judge → filter →
dedup → balance flow, but inverted seed strategy and a hard safety filter.

## What's different from the profanity pipeline

| | Bad-word pipeline | Clean pipeline |
|---|---|---|
| Seed file | `words.txt` (profanity) | `topics.txt` (traits/situations: weight, grades, accent, family...) |
| Categories | 7 (targeted insult, threat, topic opinion, venting, meta, banter) | 14 (adds exclusion, rumor-spreading, backhanded compliments, gaslighting, pile-on, genuine compliments, constructive feedback, supportive messages, neutral statements) |
| Constraint | none | **hard blocklist filter** — any generated row containing a word from `blocklist.txt` (403 words, pulled from your prior `generated_raw.csv` `source_word` column) is dropped immediately |
| Concurrency | sequential | `ThreadPoolExecutor` (6 workers) + `OLLAMA_NUM_PARALLEL=4`, tuned for RTX 5060 / 16GB RAM |
| Checkpoints | CSV every 10 words | CSV + **3 random printed samples** every 5 topics, so quality is visible mid-run |

## Why bullying-without-bad-words needs its own categories

Real-world bullying is mostly **not** slur-laden — it's exclusion, rumor, backhanded
compliments, and manipulation. A classifier only trained on profanity-adjacent bullying
will miss most real cyberbullying and over-flag any sentence with a curse word regardless
of context (the same shortcut problem as before, just inverted).

## Hard constraints enforced

1. Generation prompt explicitly forbids profanity/slurs (soft constraint — LLMs still slip).
2. Post-generation blocklist filter checks every row against 403 known bad words/phrases (hard constraint, catches the slips).
3. Final safety re-check after balancing, before export — belt and suspenders.

## Running it

```bash
export OLLAMA_NUM_PARALLEL=4
ollama pull dolphin-mistral
ollama pull qwen2.5:7b-instruct
pip install requests sentence-transformers scikit-learn pandas --break-system-packages
python pipeline_clean.py
```

## Scaling to 10k+

Current: 88 topics × 14 categories × 10/call ≈ 12,320 raw generations before
blocklist/judge/dedup losses (expect ~60-75% survival → roughly 7,500-9,000 final rows
on the first pass).

To close the gap to 10k without a second full run:
- Add more topics to `topics.txt` (each new topic ≈ 140 more raw generations)
- Bump `N_PER_CALL` from 10 → 14
- Bump `MAX_PER_TOPIC_CATEGORY` cap from 10 → 14 (less aggressive dedup-balance)

The script is idempotent-ish per run — re-running after adding topics just adds more
rows to a fresh `output_clean/` (rename the old one first if you want to keep both and
concatenate `final_dataset_clean.csv` files afterward).

## Files
- `pipeline_clean.py` — the pipeline
- `templates_clean.py` — 14 categories, topic-seeded
- `topics.txt` — 88 seed topics (add more to reach 10k faster)
- `blocklist.txt` — 403 profanity words/phrases, auto-extracted from your prior `generated_raw.csv`

## Open questions
- [ ] Run on 5-10 topics first, check `output_clean/checkpoint_generated.csv` quality before full run
- [ ] Decide whether to merge this with the bad-word dataset into one final CSV, or keep separate and combine at train time
- [ ] Manual review pass on `needs_review_clean.jsonl`
