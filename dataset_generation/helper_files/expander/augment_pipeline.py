#!/usr/bin/env python3
"""
Controlled Source-Word Augmentation Pipeline
============================================
Hierarchy (NOT sentence-first):

    unique source words -> seeded random selection of N words
        -> mutation pool per word (~30 candidates)
            -> seeded random selection of ~8 mutations
                -> 2-3 sentence contexts per mutation
                    -> replace ONLY the source word

Deterministic: same seed + same config => identical output.
"""

import json
import os
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd

# ----------------------------------------------------------------------------
# 21. CONFIGURATION
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
SOURCE_WORDS_TO_SELECT = 20
MUTATION_POOL_SIZE = 30
MUTATIONS_TO_USE_PER_WORD = 8
MIN_SENTENCES_PER_MUTATION = 2
MAX_SENTENCES_PER_MUTATION = 3
SIMILARITY_THRESHOLD = 0.50

TRANSFORMATION_DISTRIBUTION = {
    "leetspeak": 0.30,
    "deletion": 0.15,
    "insertion": 0.15,
    "repetition": 0.10,
    "keyboard_noise": 0.10,
    "mixed_script": 0.10,
    "spacing_noise": 0.10,
}

LEVEL_DISTRIBUTION = {"mild": 0.60, "moderate": 0.30, "heavy": 0.10}

INPUT_FILE = os.environ.get("AUG_INPUT", "checker_till_499.csv")
OUTPUT_DIR = os.environ.get("AUG_OUTDIR", "dataset_generation/helper_files/expander")
OUTPUT_FILE = "cyberbullying_augmented_noise.csv"
REPORT_FILE = "cyberbullying_augmentation_report.csv"
CONFIG_FILE = "augmentation_config.json"

# ----------------------------------------------------------------------------
# 5. TRANSFORMATION PRIMITIVES
# ----------------------------------------------------------------------------
LEET_MAP = {
    "o": "0", "i": "1", "l": "1", "e": "3", "a": "4",
    "s": "5", "t": "7", "b": "8", "g": "9", "z": "2",
}

# Visually similar Cyrillic / Greek homoglyphs only. No exotic garbage.
HOMOGLYPH_MAP = {
    "a": "\u0430",  # CYRILLIC SMALL A
    "c": "\u0441",  # CYRILLIC SMALL ES
    "e": "\u0435",  # CYRILLIC SMALL IE
    "o": "\u043e",  # CYRILLIC SMALL O
    "p": "\u0440",  # CYRILLIC SMALL ER
    "x": "\u0445",  # CYRILLIC SMALL HA
    "y": "\u0443",  # CYRILLIC SMALL U
    "i": "\u0456",  # CYRILLIC SMALL BYELORUSSIAN-UKRAINIAN I
    "s": "\u0455",  # CYRILLIC SMALL DZE
    "n": "\u03b7",  # GREEK SMALL ETA
}

KEYBOARD_NEIGHBOURS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
    "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
    "0": "9o", "1": "2q", "2": "13w", "3": "24e", "4": "35r", "5": "46t",
    "6": "57y", "7": "68u", "8": "79i", "9": "80o",
}

INSERTABLE = string.ascii_lowercase
VOWELS = "aeiou"

# indices of characters that are safe to mutate (letters/digits only)
def _mutable_indices(word):
    return [i for i, ch in enumerate(word) if ch.isalnum()]


def _apply_leet(word, rng, n_edits):
    idxs = [i for i, ch in enumerate(word) if ch.lower() in LEET_MAP]
    if not idxs:
        return None
    rng.shuffle(idxs)
    chars = list(word)
    for i in idxs[:n_edits]:
        chars[i] = LEET_MAP[chars[i].lower()]
    return "".join(chars)


def _apply_deletion(word, rng, n_edits):
    idxs = _mutable_indices(word)
    if len(word) - n_edits < 3 or len(idxs) <= n_edits:
        return None
    # avoid deleting the first character: keeps the word recognizable
    idxs = [i for i in idxs if i != 0] or idxs
    rng.shuffle(idxs)
    drop = set(idxs[:n_edits])
    return "".join(ch for i, ch in enumerate(word) if i not in drop)


def _apply_insertion(word, rng, n_edits):
    chars = list(word)
    for _ in range(n_edits):
        pos = rng.randint(1, len(chars))
        chars.insert(pos, rng.choice(INSERTABLE))
    return "".join(chars)


def _apply_repetition(word, rng, n_edits):
    idxs = _mutable_indices(word)
    if not idxs:
        return None
    chars = list(word)
    for _ in range(n_edits):
        i = rng.choice(_mutable_indices("".join(chars)))
        chars.insert(i, chars[i])
    return "".join(chars)


def _apply_keyboard_noise(word, rng, n_edits):
    chars = list(word)
    mode = rng.random()
    if mode < 0.65:  # adjacent-key substitution
        idxs = [i for i, ch in enumerate(chars) if ch.lower() in KEYBOARD_NEIGHBOURS]
        if not idxs:
            return None
        rng.shuffle(idxs)
        for i in idxs[:n_edits]:
            chars[i] = rng.choice(KEYBOARD_NEIGHBOURS[chars[i].lower()])
    else:  # transposition of two adjacent characters
        if len(chars) < 3:
            return None
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _apply_mixed_script(word, rng, n_edits):
    n_edits = min(n_edits, 2)  # spec: max 1-2 homoglyph substitutions
    idxs = [i for i, ch in enumerate(word) if ch.lower() in HOMOGLYPH_MAP]
    if not idxs:
        return None
    rng.shuffle(idxs)
    chars = list(word)
    for i in idxs[:n_edits]:
        chars[i] = HOMOGLYPH_MAP[chars[i].lower()]
    return "".join(chars)


def _apply_spacing_noise(word, rng, n_edits):
    if len(word) < 4:
        return None
    chars = list(word)
    sep = rng.choice([" ", ".", "-", "_", "*"])
    pos = rng.randint(1, len(chars) - 1)
    chars.insert(pos, sep)
    if n_edits > 1 and len(chars) > 5:
        pos2 = rng.randint(1, len(chars) - 1)
        chars.insert(pos2, rng.choice([" ", "."]))
    return "".join(chars)


TRANSFORMERS = {
    "leetspeak": _apply_leet,
    "deletion": _apply_deletion,
    "insertion": _apply_insertion,
    "repetition": _apply_repetition,
    "keyboard_noise": _apply_keyboard_noise,
    "mixed_script": _apply_mixed_script,
    "spacing_noise": _apply_spacing_noise,
}

# ----------------------------------------------------------------------------
# 17. QUALITY CONTROL
# ----------------------------------------------------------------------------
INVISIBLE = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u200e", "\u200f", "\u00ad",
}


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_valid_mutation(original, mutation):
    """Returns (ok: bool, reason: str)."""
    if not mutation or mutation == original:
        return False, "identical_or_empty"
    if mutation.lower() == original.lower():
        return False, "case_only_change"
    for ch in mutation:
        if ch in INVISIBLE:
            return False, "invisible_char"
        if unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs"):
            return False, "control_char"
    if similarity(original, mutation) < SIMILARITY_THRESHOLD:
        return False, "below_similarity_threshold"
    if len(mutation) < 2:
        return False, "too_short"
    if len(mutation) > len(original) * 2:
        return False, "excessively_corrupted"
    return True, "ok"


# ----------------------------------------------------------------------------
# 4 + 14. MUTATION POOL GENERATION (with level control)
# ----------------------------------------------------------------------------
def _weighted_keys(dist):
    return list(dist.keys()), list(dist.values())


def generate_mutation_pool(word, rng, pool_size=MUTATION_POOL_SIZE):
    """Generate ~pool_size unique, validated obfuscations of `word`.

    Each candidate is planned first (level -> transformation types -> edit
    count) so the pool approximates both the transformation distribution and
    the mild/moderate/heavy distribution.
    """
    t_keys, t_weights = _weighted_keys(TRANSFORMATION_DISTRIBUTION)
    l_keys, l_weights = _weighted_keys(LEVEL_DISTRIBUTION)

    pool = {}            # mutation -> (transformation_label, level)
    rejected = Counter()

    # Level quotas so the pool itself matches the 60/30/10 target; random
    # selection from the pool then inherits that distribution.
    quota = {lvl: max(1, round(LEVEL_DISTRIBUTION[lvl] * pool_size)) for lvl in l_keys}
    drift = pool_size - sum(quota.values())
    quota["mild"] += drift
    remaining = dict(quota)
    made = Counter()

    attempts = 0
    max_attempts = pool_size * 60

    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        open_levels = [l for l in l_keys if made[l] < remaining[l]]
        if not open_levels:                       # quotas met but pool short
            open_levels = l_keys
        level = rng.choices(
            open_levels, weights=[LEVEL_DISTRIBUTION[l] for l in open_levels], k=1
        )[0]
        primary = rng.choices(t_keys, weights=t_weights, k=1)[0]

        if level == "mild":
            # one single edit
            mutated = TRANSFORMERS[primary](word, rng, 1)
            label = primary
        elif level == "moderate":
            # 2-3 edits of the same transformation type
            mutated = TRANSFORMERS[primary](word, rng, rng.randint(2, 3))
            label = primary
        else:  # heavy: two transformation types combined
            secondary = rng.choices(t_keys, weights=t_weights, k=1)[0]
            while secondary == primary:
                secondary = rng.choices(t_keys, weights=t_weights, k=1)[0]
            first = TRANSFORMERS[primary](word, rng, rng.randint(1, 2))
            if first is None:
                rejected["transform_not_applicable"] += 1
                continue
            mutated = TRANSFORMERS[secondary](first, rng, 1)
            label = f"{primary}+{secondary}"

        if mutated is None:
            rejected["transform_not_applicable"] += 1
            continue

        ok, reason = is_valid_mutation(word, mutated)
        if not ok:
            rejected[reason] += 1
            continue
        if mutated in pool:                       # 16. dedup within pool
            rejected["duplicate_mutation"] += 1
            continue

        pool[mutated] = (label, level)
        made[level] += 1

    return pool, rejected


# ----------------------------------------------------------------------------
# 3 + 8. SENTENCE CONTEXT HANDLING
# ----------------------------------------------------------------------------
def find_occurrence(text, word):
    """Case-insensitive literal search. Returns (start, end) or None."""
    m = re.search(re.escape(word), text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def replace_source_word(text, word, mutation):
    """Replace ONLY the first occurrence of `word`. Everything else untouched."""
    span = find_occurrence(text, word)
    if span is None:
        return None
    s, e = span
    return text[:s] + mutation + text[e:]


# ----------------------------------------------------------------------------
# MAIN PIPELINE (22. FINAL IMPLEMENTATION FLOW)
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # --- 1. Load (original file never modified) ------------------------------
    df = pd.read_csv(INPUT_FILE)
    df = df.reset_index(drop=True)

    dataset_stats = {
        "total_rows": int(len(df)),
        "unique_source_words": int(df["source_word"].nunique()),
        "category_distribution": df["category"].value_counts().to_dict(),
        "source_word_frequency_min": int(df["source_word"].value_counts().min()),
        "source_word_frequency_max": int(df["source_word"].value_counts().max()),
        "source_word_frequency_mean": round(float(df["source_word"].value_counts().mean()), 2),
    }
    match_mask = df.apply(
        lambda r: find_occurrence(str(r["text"]), str(r["source_word"])) is not None, axis=1
    )
    dataset_stats["pct_rows_source_word_in_text"] = round(float(match_mask.mean() * 100), 2)

    # --- 18. group ids: one per ORIGINAL sentence ----------------------------
    df["augmentation_group_id"] = [f"GRP_{i:06d}" for i in range(1, len(df) + 1)]

    # --- 2. Source-word selection -------------------------------------------
    unique_words = sorted(df["source_word"].astype(str).unique())
    n_select = min(SOURCE_WORDS_TO_SELECT, len(unique_words))
    selected_words = rng.sample(unique_words, n_select)

    # --- Per-word processing -------------------------------------------------
    aug_rows = []
    mutation_stats, sentence_stats, unmatched_rows, shortfalls = [], [], [], []
    rejected_totals = Counter()
    global_texts = set(df["text"].astype(str).str.strip().str.lower())
    seen_pairs = set()          # (source_word, mutation)
    seen_group_mut = set()      # (group_id, mutation)
    duplicate_count = 0
    aug_counter = 0

    for word in selected_words:
        sub = df[df["source_word"].astype(str) == word]

        # --- 3. validation: does the source word actually occur in the text? -
        valid_ctx, invalid_ctx = [], []
        for _, row in sub.iterrows():
            if find_occurrence(str(row["text"]), word) is not None:
                valid_ctx.append(row)
            else:
                invalid_ctx.append(row)
        for row in invalid_ctx:
            unmatched_rows.append({
                "source_word": word,
                "text": row["text"],
                "reason": "source_word_not_found_in_text",
            })

        # --- 4. mutation pool ------------------------------------------------
        pool, rejected = generate_mutation_pool(word, rng)
        rejected_totals.update(rejected)

        # --- 6. seeded selection of a subset (never "the first N") -----------
        pool_items = sorted(pool.items())                    # deterministic order
        k = min(MUTATIONS_TO_USE_PER_WORD, len(pool_items))
        chosen = rng.sample(pool_items, k) if pool_items else []

        mutation_stats.append({
            "source_word": word,
            "mutation_pool_size": len(pool_items),
            "mutations_selected": len(chosen),
            "valid_contexts": len(valid_ctx),
            "unmatched_contexts": len(invalid_ctx),
        })

        if not valid_ctx:
            shortfalls.append({"source_word": word, "mutation": "", "requested": 0,
                               "available": 0, "reason": "no_valid_sentence_contexts"})
            continue

        for mutation, (transform, level) in chosen:
            if (word, mutation) in seen_pairs:                 # 16. dedup
                duplicate_count += 1
                continue
            seen_pairs.add((word, mutation))

            # --- 7. how many sentence variants for this mutation? ------------
            want = rng.randint(MIN_SENTENCES_PER_MUTATION, MAX_SENTENCES_PER_MUTATION)
            # --- 9. context diversity: sample WITHOUT replacement ------------
            avail = min(want, len(valid_ctx))
            if avail < MIN_SENTENCES_PER_MUTATION:
                shortfalls.append({"source_word": word, "mutation": mutation,
                                   "requested": want, "available": len(valid_ctx),
                                   "reason": "insufficient_contexts"})
            contexts = rng.sample(valid_ctx, avail)

            produced = 0
            for row in contexts:
                gid = row["augmentation_group_id"]
                if (gid, mutation) in seen_group_mut:
                    duplicate_count += 1
                    continue

                new_text = replace_source_word(str(row["text"]), word, mutation)
                if new_text is None:
                    continue

                key = new_text.strip().lower()
                if key in global_texts:                        # 16. global dedup
                    duplicate_count += 1
                    continue

                # --- 17. sentence-level QC ------------------------------------
                if mutation not in new_text:
                    continue

                global_texts.add(key)
                seen_group_mut.add((gid, mutation))
                aug_counter += 1
                aug_rows.append({
                    "text": new_text,
                    "gen_label": row["gen_label"],              # 11. never changed
                    "category": row["category"],
                    "target_type": row["target_type"],
                    "source_word": mutation,                    # 12.
                    "original_source_word": word,
                    "augmentation_type": transform,
                    "augmentation_level": level,
                    "original_text": row["text"],
                    "augmentation_id": f"AUG_{aug_counter:06d}",
                    "augmentation_group_id": gid,               # 18.
                    "is_augmented": 1,
                })
                produced += 1

            sentence_stats.append({
                "source_word": word,
                "mutation": mutation,
                "transformation": transform,
                "level": level,
                "sentence_variants_requested": want,
                "sentence_variants_generated": produced,
            })

    aug_df = pd.DataFrame(aug_rows)

    # --- Combine originals + augmented --------------------------------------
    orig_out = df.copy()
    orig_out["original_source_word"] = ""
    orig_out["augmentation_type"] = "original"
    orig_out["augmentation_level"] = "none"
    orig_out["original_text"] = ""
    orig_out["augmentation_id"] = ""
    orig_out["is_augmented"] = 0

    cols = ["text", "gen_label", "category", "target_type", "source_word",
            "original_source_word", "augmentation_type", "augmentation_level",
            "original_text", "augmentation_id", "augmentation_group_id", "is_augmented"]
    final = pd.concat([orig_out[cols], aug_df[cols] if len(aug_df) else pd.DataFrame(columns=cols)],
                      ignore_index=True)

    final.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILE), index=False)

    # ------------------------------------------------------------------------
    # 20. REPORT  (single long-format CSV with a `section` column)
    # ------------------------------------------------------------------------
    rep = []

    def add(section, **kw):
        rep.append({"section": section, **kw})

    add("dataset", metric="total_rows", value=dataset_stats["total_rows"])
    add("dataset", metric="unique_source_words", value=dataset_stats["unique_source_words"])
    add("dataset", metric="pct_rows_source_word_in_text",
        value=dataset_stats["pct_rows_source_word_in_text"])
    add("dataset", metric="source_word_freq_mean", value=dataset_stats["source_word_frequency_mean"])

    add("source_word_selection", metric="random_seed", value=RANDOM_SEED)
    add("source_word_selection", metric="total_unique_source_words", value=len(unique_words))
    add("source_word_selection", metric="selected_source_words",
        value="|".join(selected_words))

    for m in mutation_stats:
        add("mutation_statistics", source_word=m["source_word"],
            mutation_pool_size=m["mutation_pool_size"],
            mutations_selected=m["mutations_selected"],
            valid_contexts=m["valid_contexts"],
            unmatched_contexts=m["unmatched_contexts"])

    for s in sentence_stats:
        add("sentence_statistics", source_word=s["source_word"], mutation=s["mutation"],
            transformation=s["transformation"], augmentation_level=s["level"],
            sentence_variants_generated=s["sentence_variants_generated"])

    if len(aug_df):
        tcounts = aug_df["augmentation_type"].value_counts()
        for t, c in tcounts.items():
            add("transformation_distribution", transformation=t, count=int(c),
                percentage=round(100 * c / len(aug_df), 2))
        lcounts = aug_df["augmentation_level"].value_counts()
        for l, c in lcounts.items():
            add("level_distribution", augmentation_level=l, count=int(c),
                percentage=round(100 * c / len(aug_df), 2))

    oc = df["category"].value_counts()
    ac = aug_df["category"].value_counts() if len(aug_df) else pd.Series(dtype=int)
    for cat in oc.index:
        add("category_distribution", category=cat, original_count=int(oc.get(cat, 0)),
            augmented_count=int(ac.get(cat, 0)),
            total_count=int(oc.get(cat, 0)) + int(ac.get(cat, 0)))

    add("quality_statistics", metric="duplicate_count", value=duplicate_count)
    add("quality_statistics", metric="invalid_mutation_count", value=int(sum(rejected_totals.values())))
    for reason, c in rejected_totals.most_common():
        add("quality_statistics", metric=f"invalid_mutation:{reason}", value=int(c))
    add("quality_statistics", metric="unmatched_source_word_row_count", value=len(unmatched_rows))
    add("quality_statistics", metric="mutation_shortfall_count", value=len(shortfalls))
    add("quality_statistics", metric="augmented_rows_generated", value=len(aug_df))
    add("quality_statistics", metric="final_rows", value=len(final))

    for u in unmatched_rows[:200]:
        add("unmatched_rows", source_word=u["source_word"], text=u["text"], reason=u["reason"])
    for s in shortfalls:
        add("shortfalls", source_word=s["source_word"], mutation=s["mutation"],
            requested=s["requested"], available=s["available"], reason=s["reason"])

    if len(aug_df):
        prev_rng = random.Random(RANDOM_SEED)
        idx = prev_rng.sample(range(len(aug_df)), min(25, len(aug_df)))
        for i in idx:
            r = aug_df.iloc[i]
            add("example_preview", original_word=r["original_source_word"],
                mutated_word=r["source_word"], transformation=r["augmentation_type"],
                augmentation_level=r["augmentation_level"],
                original_sentence=r["original_text"], augmented_sentence=r["text"])

    pd.DataFrame(rep).to_csv(os.path.join(OUTPUT_DIR, REPORT_FILE), index=False)

    # --- 21. config ----------------------------------------------------------
    config = {
        "random_seed": RANDOM_SEED,
        "source_words_to_select": SOURCE_WORDS_TO_SELECT,
        "mutation_pool_size": MUTATION_POOL_SIZE,
        "mutations_to_use_per_word": MUTATIONS_TO_USE_PER_WORD,
        "min_sentences_per_mutation": MIN_SENTENCES_PER_MUTATION,
        "max_sentences_per_mutation": MAX_SENTENCES_PER_MUTATION,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "transformation_distribution": TRANSFORMATION_DISTRIBUTION,
        "level_distribution": LEVEL_DISTRIBUTION,
        "input_file": os.path.basename(INPUT_FILE),
        "output_file": OUTPUT_FILE,
        "selected_source_words": selected_words,
        "dataset_stats": dataset_stats,
        "results": {
            "augmented_rows": int(len(aug_df)),
            "original_rows": int(len(df)),
            "final_rows": int(len(final)),
            "duplicate_count": duplicate_count,
            "invalid_mutation_count": int(sum(rejected_totals.values())),
            "unmatched_source_word_row_count": len(unmatched_rows),
            "mutation_shortfall_count": len(shortfalls),
        },
    }
    with open(os.path.join(OUTPUT_DIR, CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"selected words : {selected_words}")
    print(f"augmented rows : {len(aug_df)}")
    print(f"final rows     : {len(final)}")
    print(f"duplicates     : {duplicate_count}  invalid mutations: {sum(rejected_totals.values())}")
    print(f"unmatched rows : {len(unmatched_rows)}  shortfalls: {len(shortfalls)}")
    return final, aug_df


if __name__ == "__main__":
    main()
