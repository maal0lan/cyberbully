"""
Clean (no-profanity) cyberbullying dataset pipeline.
Optimized for: RTX 5060, 16GB RAM, dolphin-mistral + qwen2.5:7b-instruct via Ollama.

Key differences from the "bad words" pipeline:
  - Seeded by TOPICS (appearance, grades, family...) not profanity.
  - Hard post-generation BLOCKLIST filter — any row containing a word from
    blocklist.txt is dropped immediately, no exceptions.
  - Concurrent generation via ThreadPoolExecutor (tuned for OLLAMA_NUM_PARALLEL=4).
  - Checkpoints every N topics: saves progress + prints a random sample so you
    can sanity-check quality without waiting for the full run.

Before running:
    export OLLAMA_NUM_PARALLEL=4
    ollama pull dolphin-mistral
    ollama pull qwen2.5:7b-instruct
    pip install requests sentence-transformers scikit-learn pandas --break-system-packages
"""

import json
import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from templates_clean import CATEGORIES, SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "dolphin-mistral"
JUDGE_MODEL = "qwen2.5:7b-instruct"

N_PER_CALL = 10             # sentences per (topic, category) call
MAX_WORKERS = 6             # python-side concurrency (tune with OLLAMA_NUM_PARALLEL)
SIM_THRESHOLD = 0.92
MAX_PER_TOPIC_CATEGORY = 10
CHECKPOINT_EVERY = 5        # topics

OUT_DIR = Path("./output_clean")
OUT_DIR.mkdir(exist_ok=True)

BLOCKLIST = [
    w.strip().lower() for w in Path("blocklist.txt").read_text().splitlines() if w.strip()
]
# sort longest-first so multi-word blocked phrases match before their substrings
BLOCKLIST.sort(key=len, reverse=True)


def contains_blocked_word(text: str) -> str | None:
    low = text.lower()
    for w in BLOCKLIST:
        if w in low:
            return w
    return None


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
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            print(f"  [warn] {model} call failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return ""


def extract_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------- Stage 1: generation (concurrent) ----------

def generate_one(topic: str, cat_name: str, cat: dict) -> list[dict]:
    instruction = cat["instruction"].format(topic=topic)
    user_prompt = f"{instruction}\n\nGenerate {N_PER_CALL} such sentences as a JSON array."
    system = SYSTEM_PROMPT.format(n=N_PER_CALL)
    raw = call_ollama(GEN_MODEL, system, user_prompt)
    parsed = extract_json(raw)
    if not parsed:
        return []

    rows = []
    blocked_count = 0
    for item in parsed:
        text = item.get("text", "").strip()
        if not text:
            continue
        hit = contains_blocked_word(text)
        if hit:
            blocked_count += 1
            continue
        rows.append({
            "text": text,
            "gen_label": int(item.get("label", cat["label"])),
            "category": cat_name,
            "target_type": item.get("target_type", "unknown"),
            "source_topic": topic,
        })
    if blocked_count:
        print(f"    [blocklist] dropped {blocked_count} rows for topic='{topic}' cat='{cat_name}'")
    return rows


def run_generation(topics: list[str]) -> pd.DataFrame:
    all_rows = []
    jobs = [(topic, cat_name, cat) for topic in topics for cat_name, cat in CATEGORIES.items()]
    print(f"Total generation calls queued: {len(jobs)} "
          f"({len(topics)} topics x {len(CATEGORIES)} categories)")

    completed_topics = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(generate_one, t, c, cat): (t, c) for t, c, cat in jobs}
        done_count = 0
        for fut in as_completed(futures):
            topic, cat_name = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [error] topic='{topic}' cat='{cat_name}': {e}")
            done_count += 1
            completed_topics.add(topic)

            if done_count % (len(CATEGORIES) * CHECKPOINT_EVERY) == 0:
                df_ckpt = pd.DataFrame(all_rows)
                df_ckpt.to_csv(OUT_DIR / "checkpoint_generated.csv", index=False)
                print(f"\n[checkpoint] {done_count}/{len(jobs)} calls done, "
                      f"{len(all_rows)} rows so far")
                if len(df_ckpt) >= 5:
                    sample = df_ckpt.sample(3, random_state=random.randint(0, 9999))
                    for _, r in sample.iterrows():
                        print(f"   ({r['category']}, label={r['gen_label']}): {r['text']}")
                print()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "generated_raw_clean.csv", index=False)
    return df


# ---------- Stage 2: judging (concurrent) ----------

JUDGE_SYSTEM = """You are a strict annotator for a cyberbullying detection dataset's \
no-profanity subset. Given a sentence (guaranteed to contain no profanity), decide if \
it is CYBERBULLYING (targeted mockery, exclusion, manipulation, clean-language insult, \
or threat aimed at a specific person/group with intent to harm) or NOT (opinions about \
topics, self-venting, genuine compliments, constructive feedback, neutral statements, \
or anti-bullying commentary).

Respond ONLY with valid JSON: {"label": 0 or 1, "reasoning": "one short sentence"}
No markdown fences, no extra text."""


def judge_one(text: str) -> dict:
    raw = call_ollama(JUDGE_MODEL, JUDGE_SYSTEM, f"Sentence: {text}")
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"label": None, "reasoning": "PARSE_FAILURE"}


def run_judging(df: pd.DataFrame) -> pd.DataFrame:
    judge_labels = [None] * len(df)
    reasonings = [""] * len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(judge_one, row.text): i for i, row in enumerate(df.itertuples())}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            result = fut.result()
            judge_labels[i] = result.get("label")
            reasonings[i] = result.get("reasoning", "")
            done += 1
            if done % 200 == 0:
                print(f"  judged {done}/{len(df)}")

    df = df.copy()
    df["judge_label"] = judge_labels
    df["judge_reasoning"] = reasonings
    return df


# ---------- Stage 3: filter ----------

def split_agreements(df: pd.DataFrame):
    df = df.dropna(subset=["judge_label"]).copy()
    df["judge_label"] = df["judge_label"].astype(int)
    agree = df[df["gen_label"] == df["judge_label"]].copy()
    disagree = df[df["gen_label"] != df["judge_label"]].copy()
    agree["label"] = agree["gen_label"]
    return agree, disagree


# ---------- Stage 4: dedup ----------

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")
    keep_idx = []
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        embeddings = model.encode(sub["text"].tolist(), show_progress_bar=False, batch_size=64)
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
    df = df.groupby(["source_topic", "category"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), MAX_PER_TOPIC_CATEGORY), random_state=42)
    )
    return df.reset_index(drop=True)


# ---------- Orchestration ----------

def main():
    topics = [t.strip() for t in Path("topics.txt").read_text().splitlines() if t.strip()]
    print(f"Loaded {len(topics)} topics, {len(CATEGORIES)} categories, "
          f"blocklist size {len(BLOCKLIST)}")
    print(f"Target raw generations: {len(topics) * len(CATEGORIES) * N_PER_CALL}")

    t0 = time.time()
    print("\n=== Stage 1: Generation ===")
    generated = run_generation(topics)
    print(f"Generated {len(generated)} rows (post-blocklist) in {time.time()-t0:.0f}s")

    print("\n=== Stage 2: Judging ===")
    judged = run_judging(generated)

    print("\n=== Stage 3: Filtering ===")
    agree, disagree = split_agreements(judged)
    print(f"Agreements: {len(agree)}  Disagreements (needs review): {len(disagree)}")
    disagree.to_json(OUT_DIR / "needs_review_clean.jsonl", orient="records", lines=True)

    print("\n=== Stage 4: Dedup ===")
    deduped = dedup(agree)
    print(f"After dedup: {len(deduped)} (from {len(agree)})")

    print("\n=== Stage 5: Balance ===")
    balanced = balance(deduped)
    print(f"After balance cap: {len(balanced)}")
    print(balanced["label"].value_counts())

    # final safety net: re-check blocklist post-everything
    final = balanced[balanced["text"].apply(lambda t: contains_blocked_word(t) is None)]

    final_cols = ["text", "label", "category", "target_type", "source_topic"]
    final[final_cols].to_csv(OUT_DIR / "final_dataset_clean.csv", index=False)
    print(f"\nDone in {time.time()-t0:.0f}s total.")
    print(f"Final dataset -> {OUT_DIR / 'final_dataset_clean.csv'} ({len(final)} rows)")
    print(f"Manual review queue -> {OUT_DIR / 'needs_review_clean.jsonl'}")

    if len(final) < 10000:
        print(f"\n[note] You have {len(final)} rows, short of the 10k target.")
        print("Add more topics to topics.txt and re-run, or raise N_PER_CALL / "
              "MAX_PER_TOPIC_CATEGORY, then run again — the script will just add more rows.")


if __name__ == "__main__":
    main()
