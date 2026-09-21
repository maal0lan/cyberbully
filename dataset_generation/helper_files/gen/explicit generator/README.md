# Cyberbullying dataset generation pipeline

See `cyberbully-dataset-pipeline.md` for the full writeup (drop into your Obsidian vault).

## Setup
```
pip install requests sentence-transformers scikit-learn pandas --break-system-packages
ollama pull dolphin-mistral
ollama pull qwen2.5:7b-instruct
```

## Usage
1. Put your ~250 words in `words.txt`, one per line (sample included).
2. `python generate_pipeline.py`
3. Manually review `output/needs_review.jsonl` (disagreements between generator and judge).
4. Final dataset lands at `output/final_dataset.csv`.

## Files
- `cyberbully-dataset-pipeline.md` — Obsidian note, full plan/reasoning
- `templates.py` — concept categories + prompts (edit to add more categories)
- `generate_pipeline.py` — the pipeline itself
- `words.txt` — your trigger words (sample of 15 — replace with your full ~250)
