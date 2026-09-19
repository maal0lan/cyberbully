"""
Cyberbullying dataset generation pipeline.

Stages:
  1. generate  -> call GEN_MODEL (dolphin-mistral) per (word, category), get JSON rows
  2. judge     -> call JUDGE_MODEL (different, stronger model) to re-label each row
  3. filter    -> keep agreements, route disagreements to needs_review.jsonl
  4. dedup     -> embedding similarity within each label class
  5. balance   -> cap rows per category so no single word/category dominates
  6. export    -> final CSV + needs_review.jsonl for manual pass

Requires:
  pip install requests sentence-transformers scikit-learn pandas --break-system-packages
  ollama pull dolphin-mistral
  ollama pull qwen2.5:7b-instruct      (or llama3.1:8b-instruct)
"""

import json
import re
import time
import requests
import pandas as pd
from pathlib import Path

from templates import CATEGORIES, SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "dolphin-mistral"
JUDGE_MODEL = "qwen2.5:7b-instruct"   # swap to llama3.1:8b-instruct if preferred

N_PER_CALL = 6            # sentences generated per (word, category) call
SIM_THRESHOLD = 0.92      # dedup cosine similarity threshold
MAX_PER_WORD_CATEGORY = 8 # cap after dedup, for balance

OUT_DIR = Path("./output")
OUT_DIR.mkdir(exist_ok=True)


# ---------- Ollama call helpers ----------

def call_ollama(model: str, system: str, user: str, retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.9},
    }
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            print(f"  [warn] {model} call failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return ""


def extract_json(raw: str):
    """Strip markdown fences / stray text, parse JSON array."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    # grab the outermost [...] if there's extra text around it
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------- Stage 1: generation ----------

def generate_for_word(word: str) -> list[dict]:
    rows = []
    for cat_name, cat in CATEGORIES.items():
        instruction = cat["instruction"].format(word=word)
        user_prompt = f"{instruction}\n\nGenerate {N_PER_CALL} such sentences as a JSON array."
        system = SYSTEM_PROMPT.format(n=N_PER_CALL)
        raw = call_ollama(GEN_MODEL, system, user_prompt)
        parsed = extract_json(raw)
        if not parsed:
            print(f"  [skip] parse failure for word='{word}' category='{cat_name}'")
            continue
        for item in parsed:
            if "text" not in item:
                continue
            rows.append({
                "text": item["text"].strip(),
                "gen_label": int(item.get("label", cat["label"])),
                "category": cat_name,
                "target_type": item.get("target_type", "unknown"),
                "source_word": word,
            })
    return rows


def run_generation(words: list[str], checkpoint_every: int = 10) -> pd.DataFrame:
    all_rows = []
    for i, word in enumerate(words, 1):
        print(f"[{i}/{len(words)}] generating for '{word}'")
        all_rows.extend(generate_for_word(word))
        if i % checkpoint_every == 0:
            pd.DataFrame(all_rows).to_csv(OUT_DIR / "checkpoint_generated.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "generated_raw.csv", index=False)
    return df


# ---------- Stage 2: judging ----------

JUDGE_SYSTEM = """You are a strict, careful annotator for a cyberbullying detection \
dataset. Given a sentence, decide if it is CYBERBULLYING (targeted at a specific \
person/group, with intent to demean, threaten, or humiliate) or NOT (opinions about \
topics, general venting, meta-commentary condemning hate, friendly banter, or anything \
without a clear human target and harmful intent).

Respond ONLY with valid JSON: {"label": 0 or 1, "reasoning": "one short sentence"}
No markdown fences, no extra text."""


def judge_row(text: str) -> dict | None:
    raw = call_ollama(JUDGE_MODEL, JUDGE_SYSTEM, f"Sentence: {text}")
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_judging(df: pd.DataFrame) -> pd.DataFrame:
    judge_labels, reasonings = [], []
    for i, row in enumerate(df.itertuples(), 1):
        if i % 25 == 0:
            print(f"  judging {i}/{len(df)}")
        result = judge_row(row.text)
        if result is None:
            judge_labels.append(None)
            reasonings.append("PARSE_FAILURE")
        else:
            judge_labels.append(result.get("label"))
            reasonings.append(result.get("reasoning", ""))
    df["judge_label"] = judge_labels
    df["judge_reasoning"] = reasonings
    return df


# ---------- Stage 3: filter agreements vs disagreements ----------

def split_agreements(df: pd.DataFrame):
    df = df.dropna(subset=["judge_label"]).copy()
    df["judge_label"] = df["judge_label"].astype(int)
    agree = df[df["gen_label"] == df["judge_label"]].copy()
    disagree = df[df["gen_label"] != df["judge_label"]].copy()
    agree["label"] = agree["gen_label"]
    return agree, disagree


# ---------- Stage 4: dedup via embeddings ----------

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    keep_idx = []
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        embeddings = model.encode(sub["text"].tolist(), show_progress_bar=False)
        kept_embeddings = []
        for idx, emb in zip(sub.index, embeddings):
            if not kept_embeddings:
                kept_embeddings.append(emb)
                keep_idx.append(idx)
                continue
            sims = cosine_similarity([emb], kept_embeddings)[0]
            if sims.max() < SIM_THRESHOLD:
                kept_embeddings.append(emb)
                keep_idx.append(idx)
    return df.loc[keep_idx].reset_index(drop=True)


# ---------- Stage 5: balance ----------

def balance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["source_word", "category"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), MAX_PER_WORD_CATEGORY), random_state=42)
    )
    return df.reset_index(drop=True)


# ---------- Orchestration ----------

def main():
    words = [w.strip() for w in Path("words.txt").read_text().splitlines() if w.strip()]
    print(f"Loaded {len(words)} words")

    print("\n=== Stage 1: Generation ===")
    generated = run_generation(words)
    print(f"Generated {len(generated)} raw rows")

    print("\n=== Stage 2: Judging ===")
    judged = run_judging(generated)

    print("\n=== Stage 3: Filtering ===")
    agree, disagree = split_agreements(judged)
    print(f"Agreements: {len(agree)}  Disagreements (needs review): {len(disagree)}")
    disagree.to_json(OUT_DIR / "needs_review.jsonl", orient="records", lines=True)

    print("\n=== Stage 4: Dedup ===")
    deduped = dedup(agree)
    print(f"After dedup: {len(deduped)} (from {len(agree)})")

    print("\n=== Stage 5: Balance ===")
    balanced = balance(deduped)
    print(f"After balance cap: {len(balanced)}")
    print(balanced["label"].value_counts())

    final_cols = ["text", "label", "category", "target_type", "source_word"]
    balanced[final_cols].to_csv(OUT_DIR / "final_dataset.csv", index=False)
    print(f"\nDone. Final dataset -> {OUT_DIR / 'final_dataset.csv'}")
    print(f"Manual review queue -> {OUT_DIR / 'needs_review.jsonl'}")


if __name__ == "__main__":
    main()
