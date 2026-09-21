"""
Cyberbullying Detection - Full Training Pipeline
=================================================
Dataset : cyberbullying_merged_dataset.csv  (13 columns, ~29k rows)
Models  : distilbert-base-uncased (default) | bert-base-uncased | microsoft/deberta-v3-base
Task    : binary (gen_label) + optional auxiliary 20-way category head (multi-task)

Install:
    pip install torch transformers scikit-learn pandas numpy tqdm demoji matplotlib
    (deberta-v3 also needs:  pip install sentencepiece tiktoken)

Usage:
    python train_cyberbully.py --smoke                      # 1-minute sanity run
    python train_cyberbully.py --prep-only                  # only clean + split, write CSVs
    python train_cyberbully.py                              # full run, DistilBERT
    python train_cyberbully.py --model microsoft/deberta-v3-base --batch-size 16 --lr 2e-5
    python train_cyberbully.py --data-paths raw.csv augmented.csv     # auto-concat + dedup
    python train_cyberbully.py --data-paths merged.csv                # or just pass merged
    python train_cyberbully.py --resume                     # continue after a crash/stop
    python train_cyberbully.py --batch-size 8 --grad-accum-steps 4    # effective batch 32, low VRAM
    python train_cyberbully.py --predict "you are pathetic" "I hate this movie"

What this script does differently from the old one (on purpose):
  1. Splits by *original sentence group* (StratifiedGroupKFold) so a sentence and its
     leetspeak/typo copies never straddle train/val/test.  It does NOT trust the
     `augmentation_group_id` column - IDs collide across the two source datasets.
  2. Derives the binary label from `category` (the spec's definition) by default, because
     the raw `gen_label` contradicts the taxonomy for ~5.3k rows.  Every changed row is
     written to relabel_review.csv so you can spot-check.  Use --label-source raw to disable.
  3. Does NOT lowercase-strip digits or symbols (that would destroy leetspeak like
     'sna7ch', '@$$h0le').  Casing is left to the tokenizer.
  4. Drops generator placeholder leaks ('[derogatory term ...]', '(person's name)').
  5. Class weights are computed from the TRAIN split, not hard-coded.
  6. Reports macro-F1, per-class P/R, PR-AUC, per-category error rates, adversarial
     robustness (is_augmented rows + fresh synthetic perturbation of clean test rows).

Patch notes (this version):
  - --data-paths accepts multiple CSVs (raw / augmented / merged, any mix); they're
    concatenated then de-duplicated by normalized sentence group, so you can hand it
    raw+augmented separately and get the same result as a pre-merged file.
  - --resume continues an interrupted run from <out-dir>/last_checkpoint.pt (model,
    optimizer, scheduler, best score, history all restored).
  - --grad-accum-steps for a larger effective batch size on small GPUs.
  - --num-workers for the dataloaders.
  - Per-epoch training now shows an ETA for the rest of the run (based on average
    epoch time so far) plus per-epoch wall time; a training_history.png (loss + macro-F1
    curves) is saved alongside the existing confusion_matrix.png.
  - No keyword/lexicon matching anywhere - explanations are meant to stay context-level
    (prob_bully + likely_category from the auxiliary head), not surface-word triggered.
"""

import os, re, sys, json, math, time, random, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
try:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        average_precision_score, confusion_matrix, classification_report,
    )
except ImportError:
    StratifiedGroupKFold = None

# Prevent transformers from loading scikit-learn when blocked by Windows Application Control
try:
    import transformers.utils.import_utils as _tui
    _tui.is_sklearn_available = lambda: False
    import transformers.utils as _tu
    _tu.is_sklearn_available = lambda: False
except Exception:
    pass



try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    def tqdm(x, **kw):
        return x

try:
    import demoji
    USE_DEMOJI = True
except ImportError:                                   # pragma: no cover
    USE_DEMOJI = False


# ============================================================================
# CONFIG DEFAULTS (all overridable from the command line)
# ============================================================================
DEFAULTS = dict(
    data_paths=[r"dataset_generation\helper_files\dataset\generated_dataset\cyberbullying_merged_dataset.csv"],   # can pass several; they get concatenated
    out_dir="./cyberbully_v0.1_run_french",
    model="distilbert-base-uncased",
    max_len=128,
    batch_size=16,
    grad_accum_steps=1,       # effective batch size = batch_size * grad_accum_steps
    epochs=4,
    lr=2e-5,
    warmup_ratio=0.10,
    weight_decay=0.01,
    patience=2,
    aux_weight=0.30,          # weight of the 20-way category loss (0 = binary only)
    label_source="category",  # "category" (recommended) | "raw"
    num_workers=0,
    seed=42,
)

# Label implied by each category (matches section 3.2 of model_training_specs.md)
CATEGORY_TO_LABEL = {
    # 0 = not cyberbullying
    "venting_no_target": 0, "topic_opinion_negative": 0, "sarcasm_among_friends": 0,
    "meta_commentary_condemning_hate": 0, "meta_commentary_condemning_bullying": 0,
    "supportive_message": 0, "neutral_statement": 0, "genuine_compliment": 0,
    "constructive_criticism": 0,
    # 1 = cyberbullying
    "threat": 1, "threat_clean": 1, "targeted_insult_identity": 1,
    "targeted_insult_general": 1, "explicit_insult_clean": 1, "backhanded_compliment": 1,
    "manipulation_gaslighting": 1, "social_exclusion": 1, "pile_on_mob": 1,
    "implicit_mockery": 1, "rumor_spreading": 1,
}
# Categories where a false positive means the model is reacting to keywords, not intent
FP_WATCH = ["sarcasm_among_friends", "meta_commentary_condemning_hate",
            "meta_commentary_condemning_bullying", "venting_no_target",
            "topic_opinion_negative"]

LABEL_NAMES = ["not_cyberbullying", "cyberbullying"]


# ############################################################################
# SECTION A - DATA PREP (no torch needed)
# ############################################################################
URL_RE      = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE  = re.compile(r"@\w+")
ALLOWED_PH  = re.compile(r"\[(?:name|username|user|url)\]|\(person'?s name\)", re.I)
BAD_PH      = re.compile(r"\[[^\]]{2,}\]|\{[^}]*\}|<[a-z ]{3,}>", re.I)
HASHTAG_RE  = re.compile(r"#(\w+)")
WS_RE       = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Transformer-friendly cleaning. Keeps punctuation, casing, digits and symbols
    (needed for adversarial/leetspeak robustness)."""
    t = str(text)
    if USE_DEMOJI:
        t = demoji.replace_with_desc(t, sep=" ")
    t = URL_RE.sub("[url]", t)
    t = ALLOWED_PH.sub("[user]", t)
    t = MENTION_RE.sub("[user]", t)
    t = HASHTAG_RE.sub(r"\1", t)
    return WS_RE.sub(" ", t).strip()


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def load_and_prepare(paths, label_source: str, out_dir: str) -> pd.DataFrame:
    """`paths` can be a single CSV path or a list of CSV paths (raw / augmented / merged,
    any combination). Files are concatenated, then de-duplicated later in this function
    by normalized sentence group, so passing e.g. [raw.csv, augmented.csv] is equivalent
    to passing a pre-merged file - and passing merged.csv alone still works as before."""
    if isinstance(paths, str):
        paths = [paths]
    frames = []
    for p in paths:
        d = pd.read_csv(p, encoding="utf-8")
        d["__source_file"] = os.path.basename(p)
        frames.append(d)
        print(f"Loaded {len(d):,} rows from {p}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    need = {"text", "gen_label", "category"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")
    n0 = len(df)
    print(f"Combined total: {n0:,} rows from {len(paths)} file(s)")

    for c, default in [("is_augmented", 0), ("original_text", np.nan),
                       ("dataset_source", "unknown"), ("target_type", "unknown")]:
        if c not in df.columns:
            df[c] = default

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["is_augmented"] = df["is_augmented"].fillna(0).astype(int)

    # -- placeholder leakage -------------------------------------------------
    stripped = df["text"].str.replace(ALLOWED_PH, "", regex=True)
    ph_mask = stripped.str.contains(BAD_PH, regex=True) | \
        df["text"].str.contains(r"\(person'?s name\)", case=False, regex=True)
    print(f"Dropping {int(ph_mask.sum()):,} rows with generator placeholders "
          f"(e.g. '[derogatory term ...]')")
    df = df[~ph_mask].copy()

    # -- labels ----------------------------------------------------------------
    df["raw_label"] = df["gen_label"].astype(int)
    if label_source == "category":
        mapped = df["category"].map(CATEGORY_TO_LABEL)
        unknown = df.loc[mapped.isna(), "category"].unique().tolist()
        if unknown:
            print(f"  ! unknown categories (keeping raw label): {unknown}")
        df["label"] = mapped.fillna(df["raw_label"]).astype(int)
        changed = df[df["label"] != df["raw_label"]]
        print(f"Label source = category. {len(changed):,} rows "
              f"({len(changed) / len(df):.1%}) differ from raw gen_label.")
        os.makedirs(out_dir, exist_ok=True)
        changed[["text", "category", "raw_label", "label"]].to_csv(
            os.path.join(out_dir, "relabel_review.csv"), index=False)
        print(f"  -> {out_dir}/relabel_review.csv  (spot-check a sample of these!)")
    else:
        df["label"] = df["raw_label"]
        print("Label source = raw gen_label  (WARNING: contradicts category taxonomy for "
              "topic_opinion_negative / meta_commentary_* in the current CSV)")

    # -- groups: one group per ORIGINAL sentence -----------------------------
    orig = np.where(df["is_augmented"] == 1,
                    df["original_text"].fillna(df["text"]), df["text"])
    df["group"] = pd.Series(orig, index=df.index).map(_norm_key)

    # -- dedup clean rows (augmented rows are kept; they share the original's group) --
    before = len(df)
    dup = (df["is_augmented"] == 0) & df.duplicated(subset=["group"], keep="first")
    df = df[~dup].copy()
    print(f"Dropped {before - len(df):,} duplicate clean sentences (normalized match)")

    # -- target_type tidy-up (82 raw values -> 5 + other) --------------------
    tt = df["target_type"].astype(str).str.strip().str.lower()
    keep = {"individual", "group", "self", "none", "topic"}
    df["target_type"] = tt.where(tt.isin(keep), "other")

    # -- clean text for the model -------------------------------------------
    df["model_text"] = df["text"].map(clean_text)
    df = df[df["model_text"] != ""].reset_index(drop=True)

    # -- category ids for the auxiliary head ---------------------------------
    cats = sorted(df["category"].unique())
    df["cat_id"] = df["category"].map({c: i for i, c in enumerate(cats)})
    df.attrs["categories"] = cats

    print(f"\nFinal usable rows: {len(df):,}  "
          f"(augmented: {int(df['is_augmented'].sum()):,})")
    vc = df["label"].value_counts().sort_index()
    print(f"Label 0: {vc.get(0, 0):,} ({vc.get(0, 0) / len(df):.1%})   "
          f"Label 1: {vc.get(1, 0):,} ({vc.get(1, 0) / len(df):.1%})")
    return df


def grouped_split(df: pd.DataFrame, seed: int):
    """80/10/10, stratified by label, grouped by original sentence."""
    s1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_i, tmp_i = next(s1.split(df, df["label"], df["group"]))
    tmp = df.iloc[tmp_i]
    s2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    va_i, te_i = next(s2.split(tmp, tmp["label"], tmp["group"]))
    train, val, test = df.iloc[tr_i], tmp.iloc[va_i], tmp.iloc[te_i]

    g_tr, g_va, g_te = set(train.group), set(val.group), set(test.group)
    assert not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te), "group leakage!"
    for n, d in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{n:5s}: {len(d):6,} rows ({len(d) / len(df):.1%})  "
              f"pos-rate={d['label'].mean():.3f}  aug-rows={int(d['is_augmented'].sum())}")
    print("Group leakage check: PASSED (no sentence group appears in two splits)")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---- synthetic adversarial perturbation (used on clean TEST rows only) ------
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEIGH = {}
for _r in _ROWS:
    for _i, _c in enumerate(_r):
        _NEIGH[_c] = [x for x in (_r[_i - 1] if _i > 0 else "", _r[_i + 1] if _i < len(_r) - 1 else "") if x]
_LEET = {"a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"],
         "s": ["$", "5"], "t": ["7"], "l": ["1"]}


def _perturb_word(w: str, mode: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    pos = rng.randrange(1, len(w) - 1)
    if mode == "leet":
        idx = [i for i, c in enumerate(w) if c.lower() in _LEET]
        if not idx:
            return w
        i = rng.choice(idx)
        return w[:i] + rng.choice(_LEET[w[i].lower()]) + w[i + 1:]
    if mode == "delete":
        return w[:pos] + w[pos + 1:]
    if mode == "insert":
        return w[:pos] + w[pos] + w[pos:]
    if mode == "keyboard":
        c = w[pos].lower()
        return w[:pos] + rng.choice(_NEIGH[c]) + w[pos + 1:] if c in _NEIGH else w
    if mode == "space":
        return w[:pos] + " " + w[pos:]
    return w


def perturb_text(text: str, rng: random.Random) -> str:
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not idx:
        return text
    mode = rng.choice(["leet", "delete", "insert", "keyboard", "space"])
    for i in rng.sample(idx, max(1, int(len(idx) * 0.25))):
        words[i] = _perturb_word(words[i], mode, rng)
    return " ".join(words)


# ############################################################################
# SECTION B - MODEL / TRAINING (torch)
# ############################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel, AutoConfig,
                          get_linear_schedule_with_warmup)


class TextDataset(Dataset):
    def __init__(self, enc, labels, cat_ids):
        self.enc, self.labels, self.cat_ids = enc, labels, cat_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i], "cat_ids": self.cat_ids[i]}


def make_collate(tokenizer):
    def collate(batch):
        padded = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "attention_mask": [b["attention_mask"] for b in batch]},
            return_tensors="pt")
        padded["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        padded["cat_ids"] = torch.tensor([b["cat_ids"] for b in batch], dtype=torch.long)
        return padded
    return collate


def make_loader(df, tokenizer, max_len, batch_size, shuffle, text_col="model_text", num_workers=0):
    enc = tokenizer(df[text_col].tolist(), truncation=True, max_length=max_len,
                    padding=False)
    ds = TextDataset(enc, df["label"].tolist(), df["cat_id"].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(tokenizer), num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


class MultiTaskClassifier(nn.Module):
    """Encoder + mean pooling + binary head + (optional) category head."""

    def __init__(self, model_name_or_dir, n_cats, dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name_or_dir)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name_or_dir))
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.bin_head = nn.Linear(h, 2)
        self.cat_head = nn.Linear(h, n_cats)

    def forward(self, input_ids, attention_mask):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        pooled = self.drop(pooled)
        return self.bin_head(pooled), self.cat_head(pooled)


def get_device_and_amp():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Device: cuda ({torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB) | AMP: {amp}")
    else:
        dev, amp = torch.device("cpu"), None
        print("Device: cpu (training will be slow; use --model distilbert-base-uncased)")
    return dev, amp


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, bin_loss, cat_loss, aux_w, opt, sched, dev, amp, scaler,
                    grad_accum_steps=1):
    model.train()
    losses, preds, gold = [], [], []
    n_batches = len(loader)
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  train", leave=False)
    for step, b in enumerate(pbar):
        ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        y, c = b["labels"].to(dev), b["cat_ids"].to(dev)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            lb, lc = model(ids, mask)
        loss = bin_loss(lb.float(), y)
        if aux_w > 0:
            loss = loss + aux_w * cat_loss(lc.float(), c)
        loss_to_log = loss.item()
        loss = loss / grad_accum_steps                       # normalize for accumulation
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_batch = (step + 1) == n_batches
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        losses.append(loss_to_log)
        preds.extend(lb.argmax(-1).cpu().tolist()); gold.extend(y.cpu().tolist())
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
    return float(np.mean(losses)), f1_score(gold, preds, average="macro", zero_division=0)


@torch.no_grad()
def predict_probs(model, loader, dev, amp):
    model.eval()
    probs = []
    for b in tqdm(loader, desc="  eval ", leave=False):
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            lb, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        probs.extend(torch.softmax(lb.float(), -1)[:, 1].cpu().tolist())
    return np.array(probs)


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def binary_report(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "threshold": thr,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "precision_bully": precision_score(y, pred, zero_division=0),
        "recall_bully": recall_score(y, pred, zero_division=0),
        "precision_not_bully": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_not_bully": recall_score(y, pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def print_report(title, r):
    print(f"\n{'=' * 60}\n{title}  (threshold={r['threshold']:.2f})\n{'=' * 60}")
    print(f"  Accuracy          : {r['accuracy']:.4f}")
    print(f"  Macro F1          : {r['macro_f1']:.4f}   <- primary metric")
    print(f"  Bully  P / R      : {r['precision_bully']:.4f} / {r['recall_bully']:.4f}")
    print(f"  Not-bully P / R   : {r['precision_not_bully']:.4f} / {r['recall_not_bully']:.4f}")
    print(f"  ROC-AUC / PR-AUC  : {r['roc_auc']:.4f} / {r['pr_auc']:.4f}")
    cm = r["confusion_matrix"]
    print(f"  Confusion [[TN FP],[FN TP]]: {cm}")


def save_confusion_png(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["not", "bully"]); ax.set_yticklabels(["not", "bully"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Test confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_history_png(history, path):
    """Plot train loss / train macro-F1 / val macro-F1 across epochs."""
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ep, [h["train_loss"] for h in history], marker="o", color="#e07b54")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss")
    ax1.grid(alpha=0.3)
    ax2.plot(ep, [h["train_macro_f1"] for h in history], marker="o", label="train macro-F1",
             color="#2a9d8f")
    ax2.plot(ep, [h["val_macro_f1"] for h in history], marker="o", label="val macro-F1",
             color="#e76f51")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.set_title("Macro F1 per epoch")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---- inference ---------------------------------------------------------------
def load_for_inference(run_dir):
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    dev, amp = get_device_and_amp()
    tok = AutoTokenizer.from_pretrained(os.path.join(run_dir, "tokenizer"))
    model = MultiTaskClassifier(os.path.join(run_dir, "encoder_config"),
                                len(cfg["categories"]), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu"))
    return model.to(dev).eval(), tok, cfg, dev, amp


@torch.no_grad()
def predict_texts(texts, run_dir):
    model, tok, cfg, dev, amp = load_for_inference(run_dir)
    out = []
    for t in texts:
        enc = tok(clean_text(t), truncation=True, max_length=cfg["max_len"], return_tensors="pt")
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            lb, lc = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        p = float(torch.softmax(lb.float(), -1)[0, 1])
        cat = cfg["categories"][int(lc.argmax(-1))]
        out.append({"text": t, "prob_bully": round(p, 4),
                    "label": LABEL_NAMES[int(p >= cfg["threshold"])], "likely_category": cat})
    return out


CONTRAST_PROBES = [
    # same surface topic, opposite intent - a keyword-shortcut model gets these wrong
    ("I hate football, it's the most boring sport ever.",            0),
    ("You're so worthless, nobody would even notice if you left.",    1),
    ("Ugh, I hate Mondays so much.",                                  0),
    ("Everyone in the group chat agreed to stop talking to you.",     1),
    ("Calling people idiots online is hateful, we should be better.", 0),
    ("You're an idiot and everyone here knows it.",                   1),
    ("Wow, you're so brave for wearing that outfit in public.",       1),
    ("Great job on the presentation, you really nailed it!",          0),
    ("That movie was garbage, the plot made no sense.",               0),
    ("I heard you cheated on your exam, everyone's talking about it.", 1),
]


# ############################################################################
# MAIN
# ############################################################################
def parse_args():
    p = argparse.ArgumentParser(description="Cyberbullying model training pipeline")
    p.add_argument("--data-paths", nargs="+", default=DEFAULTS["data_paths"],
                   help="one or more CSVs (raw / augmented / merged) - they get concatenated "
                        "and de-duplicated by normalized sentence group automatically")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum_steps"],
                   help="accumulate gradients over N steps; effective batch = batch_size * N")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--aux-weight", type=float, default=DEFAULTS["aux_weight"],
                   help="weight of the category head loss; 0 = binary only")
    p.add_argument("--label-source", choices=["category", "raw"], default=DEFAULTS["label_source"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--prep-only", action="store_true", help="clean+split, write CSVs, stop")
    p.add_argument("--smoke", action="store_true", help="tiny fast run to verify the setup")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out-dir>/last_checkpoint.pt if it exists")
    p.add_argument("--predict", nargs="+", metavar="TEXT",
                   help="load the model in --out-dir and classify these texts")
    return p.parse_args()


def main():
    a = parse_args()

    if a.predict:
        for r in predict_texts(a.predict, a.out_dir):
            print(f"[{r['prob_bully']:.3f}] {r['label']:18s} ({r['likely_category']})  {r['text']}")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    set_seed(a.seed)

    # ---- 1. data -------------------------------------------------------------
    df = load_and_prepare(a.data_paths, a.label_source, a.out_dir)
    if a.smoke:
        df = df.sample(min(1500, len(df)), random_state=a.seed).reset_index(drop=True)
        a.epochs, a.patience = 1, 1
        print(f"\n[SMOKE MODE] using {len(df)} rows, 1 epoch")
    categories = df.attrs.get("categories") or sorted(df["category"].unique())
    print("\nSplitting (grouped + stratified)...")
    train, val, test = grouped_split(df, a.seed)

    keep_cols = ["text", "label", "raw_label", "category", "target_type",
                 "dataset_source", "is_augmented", "group"]
    for n, d in [("train", train), ("val", val), ("test", test)]:
        d[keep_cols].to_csv(os.path.join(a.out_dir, f"{n}.csv"), index=False)
    print(f"Saved train/val/test CSVs -> {a.out_dir}/")
    if a.prep_only:
        return

    # ---- 2. model, tokenizer, loaders ----------------------------------------
    dev, amp = get_device_and_amp()
    print(f"\nLoading tokenizer + model: {a.model}")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = MultiTaskClassifier(a.model, len(categories)).to(dev)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tr_loader = make_loader(train, tokenizer, a.max_len, a.batch_size, True, num_workers=a.num_workers)
    va_loader = make_loader(val, tokenizer, a.max_len, a.batch_size * 2, False, num_workers=a.num_workers)
    te_loader = make_loader(test, tokenizer, a.max_len, a.batch_size * 2, False, num_workers=a.num_workers)

    # class weights from the TRAIN split:  w_c = N / (K * N_c)
    n = len(train); counts = train["label"].value_counts().to_dict()
    cw = [n / (2 * counts.get(c, 1)) for c in (0, 1)]
    print(f"Class weights (from train): not_bully={cw[0]:.4f}  bully={cw[1]:.4f}")
    bin_loss = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=dev))
    cat_loss = nn.CrossEntropyLoss()

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for nme, p in model.named_parameters()
                    if not any(nd in nme for nd in no_decay)], "weight_decay": a.weight_decay},
        {"params": [p for nme, p in model.named_parameters()
                    if any(nd in nme for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = AdamW(groups, lr=a.lr)
    total = len(tr_loader) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * a.warmup_ratio), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp == torch.float16))

    # ---- 3. training loop ------------------------------------------------------
    ckpt = os.path.join(a.out_dir, "best_model.pt")
    resume_ckpt = os.path.join(a.out_dir, "last_checkpoint.pt")
    best_f1, best_thr, bad, history, start_ep = -1.0, 0.5, 0, [], 1

    if a.resume and os.path.exists(resume_ckpt):
        state = torch.load(resume_ckpt, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        best_f1, best_thr, bad = state["best_f1"], state["best_thr"], state["bad"]
        history, start_ep = state["history"], state["epoch"] + 1
        print(f"\nResumed from {resume_ckpt} -> starting at epoch {start_ep} "
              f"(best val macroF1 so far = {best_f1:.4f})")

    yv = val["label"].values
    t0 = time.time()
    epoch_durations = []
    for ep in range(start_ep, a.epochs + 1):
        ep_t0 = time.time()
        # ETA across remaining epochs, based on the average epoch time so far
        if epoch_durations:
            avg_ep = sum(epoch_durations) / len(epoch_durations)
            remaining = avg_ep * (a.epochs - ep + 1)
            eta_str = f" | ETA for run: {remaining / 60:.1f} min ({remaining:.0f}s)"
        else:
            eta_str = ""
        print(f"\n-- Epoch {ep}/{a.epochs}{eta_str} --")
        tr_loss, tr_f1 = train_one_epoch(model, tr_loader, bin_loss, cat_loss, a.aux_weight,
                                         opt, sched, dev, amp, scaler,
                                         grad_accum_steps=a.grad_accum_steps)
        pv = predict_probs(model, va_loader, dev, amp)
        v_f1_05 = f1_score(yv, (pv >= 0.5).astype(int), average="macro", zero_division=0)
        v_acc = accuracy_score(yv, (pv >= 0.5).astype(int))
        ep_dur = time.time() - ep_t0
        epoch_durations.append(ep_dur)
        print(f"  train loss={tr_loss:.4f} macroF1={tr_f1:.4f} | "
              f"val macroF1={v_f1_05:.4f} acc={v_acc:.4f} | epoch time={ep_dur:.1f}s")
        history.append({"epoch": ep, "train_loss": tr_loss, "train_macro_f1": tr_f1,
                        "val_macro_f1": v_f1_05, "val_acc": v_acc, "epoch_seconds": ep_dur})
        if v_f1_05 > best_f1:
            best_f1, bad = v_f1_05, 0
            best_thr, _ = best_threshold(yv, pv)
            torch.save(model.state_dict(), ckpt)
            print(f"  * new best val macroF1={best_f1:.4f}  (tuned thr={best_thr:.2f}) -> saved")
        else:
            bad += 1
            print(f"  no improvement ({bad}/{a.patience})")

        # always save a resumable checkpoint (separate from the "best" one) after every epoch
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "best_f1": best_f1, "best_thr": best_thr,
                    "bad": bad, "history": history, "epoch": ep}, resume_ckpt)

        if bad >= a.patience:
            print("  early stopping")
            break
    total_min = (time.time() - t0) / 60
    print(f"\nTraining time this run: {total_min:.1f} min "
          f"(avg {np.mean(epoch_durations):.1f}s/epoch over {len(epoch_durations)} epoch(s))")
    save_history_png(history, os.path.join(a.out_dir, "training_history.png"))

    # ---- 4. final evaluation on TEST -------------------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    pt = predict_probs(model, te_loader, dev, amp)
    yt = test["label"].values

    rep_default = binary_report(yt, pt, 0.5)
    rep_tuned = binary_report(yt, pt, best_thr)
    print_report("TEST - default threshold", rep_default)
    print_report("TEST - threshold tuned on VAL", rep_tuned)
    print("\n" + classification_report(yt, (pt >= best_thr).astype(int), digits=4,
                                       target_names=LABEL_NAMES, zero_division=0))
    save_confusion_png(rep_tuned["confusion_matrix"], os.path.join(a.out_dir, "confusion_matrix.png"))

    pred_t = (pt >= best_thr).astype(int)
    tdf = test.copy(); tdf["prob_bully"] = pt; tdf["pred"] = pred_t
    tdf["correct"] = (tdf["pred"] == tdf["label"]).astype(int)

    # per-category error (label-0 categories -> false-positive rate; label-1 -> miss rate)
    rows = []
    for cat, g in tdf.groupby("category"):
        lab = int(g["label"].iloc[0]) if g["label"].nunique() == 1 else int(round(g["label"].mean()))
        rows.append({"category": cat, "n": len(g), "expected_label": lab,
                     "error_rate": 1 - g["correct"].mean(),
                     "error_type": "false_positive" if lab == 0 else "false_negative",
                     "mean_prob_bully": g["prob_bully"].mean()})
    cat_df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    cat_df.to_csv(os.path.join(a.out_dir, "per_category_errors.csv"), index=False)
    print("\nPer-category error rates (worst first):")
    print(cat_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFalse-positive watch-list (keyword-shortcut indicators):")
    for c in FP_WATCH:
        r = cat_df[cat_df["category"] == c]
        if len(r):
            print(f"  {c:38s} FP rate = {r['error_rate'].iloc[0]:.3f}  (n={int(r['n'].iloc[0])})")

    # adversarial robustness
    adv = {}
    aug = tdf[tdf["is_augmented"] == 1]
    if len(aug):
        adv["augmented_rows_n"] = int(len(aug))
        adv["augmented_rows_accuracy"] = float(aug["correct"].mean())
        print(f"\nAdversarial (pre-made is_augmented rows in test): "
              f"n={len(aug)}  accuracy={adv['augmented_rows_accuracy']:.4f}")
    clean = test[test["is_augmented"] == 0].reset_index(drop=True)
    rng = random.Random(a.seed)
    pert = clean.copy()
    pert["model_text"] = [clean_text(perturb_text(t, rng)) for t in clean["text"]]
    pp = predict_probs(model, make_loader(pert, tokenizer, a.max_len, a.batch_size * 2, False,
                                          num_workers=a.num_workers), dev, amp)
    pc = predict_probs(model, make_loader(clean, tokenizer, a.max_len, a.batch_size * 2, False,
                                          num_workers=a.num_workers), dev, amp)
    yc = clean["label"].values
    adv["clean_accuracy"] = float(accuracy_score(yc, (pc >= best_thr).astype(int)))
    adv["perturbed_accuracy"] = float(accuracy_score(yc, (pp >= best_thr).astype(int)))
    adv["clean_macro_f1"] = float(f1_score(yc, (pc >= best_thr).astype(int), average="macro"))
    adv["perturbed_macro_f1"] = float(f1_score(yc, (pp >= best_thr).astype(int), average="macro"))
    adv["prediction_flip_rate"] = float(np.mean((pc >= best_thr) != (pp >= best_thr)))
    print(f"Synthetic perturbation of clean test rows (leet/typo/spacing): "
          f"acc {adv['clean_accuracy']:.4f} -> {adv['perturbed_accuracy']:.4f} | "
          f"macroF1 {adv['clean_macro_f1']:.4f} -> {adv['perturbed_macro_f1']:.4f} | "
          f"flip rate {adv['prediction_flip_rate']:.3f}")

    tdf.drop(columns=["group"]).to_csv(os.path.join(a.out_dir, "test_predictions.csv"), index=False)

    # ---- 5. save everything ------------------------------------------------------
    tokenizer.save_pretrained(os.path.join(a.out_dir, "tokenizer"))
    model.encoder.config.save_pretrained(os.path.join(a.out_dir, "encoder_config"))
    with open(os.path.join(a.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "max_len": a.max_len, "threshold": best_thr,
                   "categories": categories, "class_weights": cw,
                   "aux_weight": a.aux_weight, "label_source": a.label_source,
                   "args": vars(a)}, f, indent=2)
    with open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test_default_threshold": rep_default, "test_tuned_threshold": rep_tuned,
                   "adversarial": adv, "history": history}, f, indent=2)
    print(f"\nSaved model + reports -> {a.out_dir}/")

    # ---- 6. contrast-pair probe ----------------------------------------------------
    print("\n-- Contrast probe (same topics, opposite intent) --")
    model.eval()
    hits = 0
    for text, gold in CONTRAST_PROBES:
        enc = tokenizer(clean_text(text), truncation=True, max_length=a.max_len, return_tensors="pt")
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            lb, _ = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        p = float(torch.softmax(lb.float(), -1)[0, 1])
        ok = int(p >= best_thr) == gold
        hits += ok
        print(f"  {'OK ' if ok else 'BAD'} [{p:.3f}] gold={gold}  {text}")
    print(f"  {hits}/{len(CONTRAST_PROBES)} correct")
    print("\nDone.")


if __name__ == "__main__":
    main()