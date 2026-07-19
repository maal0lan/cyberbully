"""
Cyberbullying Detection — Final Version
========================================
Dataset : cyberbullying_data.csv
Model   : DistilBERT-base-uncased
Target  : 97–98 % accuracy

To squeeze out +1–2 % more: change MODEL_NAME to 'bert-base-uncased' (slower but stronger)

Run:
    pip install transformers torch scikit-learn pandas tqdm demoji
    python cyberbully_final.py
"""

import os, re, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONUTF8"]            = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silence Windows fork warning

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

try:
    import demoji
    USE_DEMOJI = True
except ImportError:
    USE_DEMOJI = False
    print("demoji not installed — emojis will be stripped. pip install demoji to keep them.")

# ══════════════════════════════════════════════════
# CONFIG  ← only change things here
# ══════════════════════════════════════════════════
DATA_PATH   = "cyberbullying_data.csv"
SAVE_DIR    = "./cyberbully_model_final"
MODEL_NAME  = "distilbert-base-uncased"   # swap → "bert-base-uncased" for +1-2 %
MAX_LEN     = 128
BATCH_SIZE  = 16     # reduce to 8 if GPU runs out of memory
EPOCHS      = 5
LR          = 2e-5
WARMUP_RATIO= 0.10
WEIGHT_DECAY= 0.01
PATIENCE    = 3      # early-stop if val F1 doesn't improve for this many epochs
NUM_WORKERS = 0      # keep 0 on Windows

# ══════════════════════════════════════════════════
# REPRODUCIBILITY + DEVICE
# ══════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEM   = DEVICE.type == "cuda"
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ══════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    BERT-aware cleaning:
    - Keeps punctuation (BERT uses it for subword tokenisation)
    - Emojis → text description (😡 → 'enraged face') — preserves signal
    - URLs / @mentions → placeholder tokens
    """
    text = str(text)
    if USE_DEMOJI:
        text = demoji.replace_with_desc(text, sep=" ")
    else:
        # fallback: strip emojis
        text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[url]",  text)
    text = re.sub(r"@\w+",                    "[user]", text)
    text = re.sub(r"#(\w+)",                  r"\1",    text)
    text = re.sub(r"\d+",                     " ",      text)
    text = re.sub(r"\s+",                     " ",      text).strip()
    return text


# ══════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════
def load_data(path: str):
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    print(f"\nLoaded {len(df):,} rows")

    # Auto-detect text & label columns
    text_col = next(
        (c for c in df.columns if any(k in c.lower()
         for k in ["text", "tweet", "comment", "message"])),
        df.columns[0]
    )
    label_col = next(
        (c for c in df.columns if any(k in c.lower()
         for k in ["label", "class", "type", "cyberbullying", "target"])),
        df.columns[1]
    )
    print(f"Text col: {text_col}  |  Label col: {label_col}")

    df = df[[text_col, label_col]].dropna()

    def to_binary(x):
        s = str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if s in {"notcyberbullying", "notbullying", "nonbullying", "normal", "neutral", "0"}:
            return 0
        if s in {"cyberbully", "cyberbullying", "bully", "bullying",
                 "abuse", "abusive", "harassment", "offensive", "toxic", "1"}:
            return 1
        if "not" in s or "non" in s: return 0
        if "bully" in s or "abuse" in s or "toxic" in s: return 1
        return float("nan")

    df["label"] = df[label_col].apply(to_binary)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"]  = df[text_col].apply(clean_text)
    df = df[df["text"].str.strip() != ""]

    print("\nClass distribution after mapping:")
    print(df["label"].value_counts().rename({0: "not_cyberbullying", 1: "cyberbullying"}))
    return df["text"].tolist(), df["label"].tolist()


# ══════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════
class BullyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════
# TRAIN / EVAL FUNCTIONS
# ══════════════════════════════════════════════════
def train_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses, preds_all, labels_all = [], [], []

    for batch in tqdm(loader, desc="  train", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)

        # ← THE CRITICAL FIX: use loss_fn(logits, labels) with class weights
        #   NOT outputs.loss which ignores class weights entirely
        loss = loss_fn(out.logits, lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        preds_all.extend(out.logits.argmax(dim=-1).cpu().tolist())
        labels_all.extend(lbl.cpu().tolist())

    return np.mean(losses), f1_score(labels_all, preds_all, zero_division=0)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    losses, probs_all, labels_all = [], [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)

        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, lbl)
        losses.append(loss.item())

        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        probs_all.extend(probs.cpu().tolist())
        labels_all.extend(lbl.cpu().tolist())

    probs_all  = np.array(probs_all)
    labels_all = np.array(labels_all)
    preds      = (probs_all >= 0.5).astype(int)

    return {
        "loss"  : float(np.mean(losses)),
        "f1"    : float(f1_score(labels_all, preds, zero_division=0)),
        "acc"   : float(accuracy_score(labels_all, preds)),
        "auc"   : float(roc_auc_score(labels_all, probs_all)),
        "probs" : probs_all,
        "labels": labels_all,
    }


# ══════════════════════════════════════════════════
# INFERENCE  (use after training or after loading)
# ══════════════════════════════════════════════════
@torch.no_grad()
def predict(text: str, model, tokenizer, threshold: float, device,
            max_len: int = MAX_LEN) -> dict:
    model.eval()
    enc  = tokenizer(
        clean_text(text),
        truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    out  = model(input_ids=enc["input_ids"].to(device),
                 attention_mask=enc["attention_mask"].to(device))
    prob = float(torch.softmax(out.logits, dim=-1)[0][1].cpu())
    pred = int(prob >= threshold)
    return {
        "text"      : text,
        "label"     : "cyberbullying" if pred == 1 else "not_cyberbullying",
        "probability": round(prob, 4),
        "threshold" : round(threshold, 4),
    }


def load_saved_model(save_dir: str = SAVE_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DistilBertForSequenceClassification.from_pretrained(save_dir).to(device)
    tok    = DistilBertTokenizerFast.from_pretrained(save_dir)
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    model.eval()
    print(f"Loaded from {save_dir}  |  threshold={cfg['threshold']:.4f}")
    return model, tok, cfg["threshold"]


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Load & clean data ─────────────────────
    texts, labels = load_data(DATA_PATH)

    # ── 2. Class weights (fixes the 5:1 imbalance) ──
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    class_weights = {int(k): float(v) for k, v in zip([0, 1], w)}
    print(f"\nClass weights: {class_weights}")
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([w[0], w[1]], dtype=torch.float).to(DEVICE)
    )

    # ── 3. Split ─────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")

    # ── 4. Tokenizer + loaders ───────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def make_loader(X, y, shuffle):
        return DataLoader(
            BullyDataset(X, y, tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEM,
        )

    train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)
    test_loader  = make_loader(X_te,  y_te,  shuffle=False)

    # ── 5. Model ─────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    no_decay = ["bias", "LayerNorm.weight"]
    params   = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer   = AdamW(params, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(total_steps * WARMUP_RATIO),
        num_training_steps= total_steps,
    )

    # ── 6. Training loop ─────────────────────────
    best_f1, patience_ctr, best_threshold = 0.0, 0, 0.5

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──")
        tr_loss, tr_f1 = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, DEVICE)
        val = eval_epoch(model, val_loader, loss_fn, DEVICE)

        print(f"  Train  loss={tr_loss:.4f}  F1={tr_f1:.4f}")
        print(f"  Val    loss={val['loss']:.4f}  F1={val['f1']:.4f}"
              f"  acc={val['acc']:.4f}  AUC={val['auc']:.4f}")

        if val["f1"] > best_f1:
            best_f1      = val["f1"]
            patience_ctr = 0

            prec, rec, thr = precision_recall_curve(val["labels"], val["probs"])
            f1s            = 2 * prec * rec / (prec + rec + 1e-9)
            best_threshold = float(thr[np.argmax(f1s)])

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  ✓ Best val F1={best_f1:.4f}  threshold={best_threshold:.4f}  → saved")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print("  Early stopping triggered.")
                break

    # ── 7. Evaluate on test set ──────────────────
    print("\nLoading best checkpoint...")
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE))

    test = eval_epoch(model, test_loader, loss_fn, DEVICE)
    te_preds_tuned   = (test["probs"] >= best_threshold).astype(int)
    te_preds_default = (test["probs"] >= 0.5).astype(int)

    for label, preds in [("0.50 (default)", te_preds_default),
                         (f"{best_threshold:.4f} (tuned)", te_preds_tuned)]:
        print(f"\n{'='*55}")
        print(f"TEST  [threshold = {label}]")
        print(f"  Accuracy : {accuracy_score(test['labels'], preds):.4f}")
        print(f"  Precision: {precision_score(test['labels'], preds, zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(test['labels'], preds, zero_division=0):.4f}")
        print(f"  F1-score : {f1_score(test['labels'], preds, zero_division=0):.4f}")
        print(f"  ROC-AUC  : {test['auc']:.4f}")
        print(confusion_matrix(test["labels"], preds))
        print(classification_report(test["labels"], preds, digits=4, zero_division=0,
                                    target_names=["not_cyberbullying", "cyberbullying"]))

    # ── 8. Save everything ───────────────────────
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump({
            "threshold"    : best_threshold,
            "model"        : MODEL_NAME,
            "class_weights": class_weights,
            "max_len"      : MAX_LEN,
        }, f, indent=2)
    print(f"\nModel saved → {SAVE_DIR}/")

    # ── 9. Quick prediction demo ─────────────────
    print("\n── Sample predictions ──")
    samples = [
        "You are useless and nobody likes you",
        "Have a wonderful day, friend!",
        "Kill yourself nobody wants you here",
        "Thanks so much for your help today",
        "Everyone hates you, just disappear",
        "Great work, keep it up!",
    ]
    for s in samples:
        r    = predict(s, model, tokenizer, best_threshold, DEVICE)
        flag = "🚨" if r["label"] == "cyberbullying" else "✅"
        print(f"  {flag}  [{r['probability']:.4f}]  {r['text']}")


if __name__ == "__main__":
    main()
