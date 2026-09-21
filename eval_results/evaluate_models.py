"""Evaluate the saved PyTorch cyberbullying model.

Run from the repository root:
    python eval_results/evaluate_models.py

Uses cyberbully_v0.1_run/test.csv and best_model.pt by default. TensorFlow and
.keras files are intentionally not used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "cyberbully_v0.1_run"
DEFAULT_OUTPUT = ROOT / "eval_results"
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
ALLOWED_PH = re.compile(r"\[(?:name|username|user|url)\]|\(person'?s name\)", re.I)
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the saved PyTorch model")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = URL_RE.sub("[url]", str(text))
    text = ALLOWED_PH.sub("[user]", text)
    text = MENTION_RE.sub("[user]", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()


class MultiTaskClassifier(nn.Module):
    def __init__(self, config_dir: Path, category_count: int):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config_dir))
        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bin_head = nn.Linear(hidden_size, 2)
        self.cat_head = nn.Linear(hidden_size, category_count)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.drop(pooled)
        return self.bin_head(pooled), self.cat_head(pooled)


def load_model(run_dir: Path, device: torch.device):
    with open(run_dir / "run_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = MultiTaskClassifier(run_dir / "encoder_config", len(config["categories"]))
    state = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, config


@torch.no_grad()
def predict_probabilities(model, tokenizer, texts, max_len, batch_size, device):
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probabilities.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.asarray(probabilities)


def threshold_table(y_true, probabilities):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, prediction),
            "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
            "f1_bully": f1_score(y_true, prediction, zero_division=0),
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
        })
    return pd.DataFrame(rows)


def save_plots(y_true, probabilities, thresholds, output, name):
    prefix = output / name
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{prefix}_roc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, probabilities):.4f}")
    ax.axhline(y_true.mean(), linestyle="--", color="gray", label=f"positive rate = {y_true.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{prefix}_pr_auc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ["accuracy", "macro_f1", "precision", "recall"]:
        ax.plot(thresholds["threshold"], thresholds[column], label=column)
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Metrics versus threshold", ylim=(0, 1.02))
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{prefix}_accuracy_vs_threshold.png", dpi=160); plt.close(fig)

    best = thresholds.loc[thresholds["macro_f1"].idxmax(), "threshold"]
    for threshold, suffix in [(0.5, "default"), (best, "best_macro_f1")]:
        matrix = confusion_matrix(y_true, (probabilities >= threshold).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not bullying", "bullying"],
               yticklabels=["not bullying", "bullying"], xlabel="Predicted", ylabel="Actual",
               title=f"Confusion matrix (threshold={threshold:.2f})")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        fig.tight_layout(); fig.savefig(f"{prefix}_confusion_matrix_{suffix}.png", dpi=160); plt.close(fig)

    true_fraction, predicted_fraction = calibration_curve(y_true, probabilities, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted_fraction, true_fraction, marker="o", label="model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction positive", title="Calibration curve")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{prefix}_calibration_curve.png", dpi=160); plt.close(fig)


def main():
    args = parse_args()
    data_path = args.data or args.run_dir / "test.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    missing = {"text", "label"} - set(data.columns)
    if missing:
        raise SystemExit(f"Data CSV is missing required columns: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(args.run_dir, device)
    texts = data["text"].fillna("").map(clean_text).tolist()
    y_true = data["label"].astype(int).to_numpy()
    probabilities = predict_probabilities(model, tokenizer, texts, config["max_len"], args.batch_size, device)
    thresholds = threshold_table(y_true, probabilities)
    best_threshold = float(thresholds.loc[thresholds["macro_f1"].idxmax(), "threshold"])
    threshold = args.threshold if args.threshold is not None else best_threshold
    prediction = (probabilities >= threshold).astype(int)
    name = "pytorch_best_model"

    thresholds.to_csv(args.output_dir / f"{name}_threshold_metrics.csv", index=False)
    predictions = data.copy()
    predictions["probability_bullying"] = probabilities
    predictions["prediction"] = prediction
    predictions["correct"] = (prediction == y_true).astype(int)
    predictions.to_csv(args.output_dir / f"{name}_predictions.csv", index=False)
    save_plots(y_true, probabilities, thresholds, args.output_dir, name)

    metrics = {
        "framework": "PyTorch",
        "model": str(args.run_dir / "best_model.pt"),
        "data": str(data_path),
        "rows": len(data),
        "device": str(device),
        "threshold": threshold,
        "best_macro_f1_threshold": best_threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, prediction, target_names=["not_cyberbullying", "cyberbullying"],
            output_dict=True, zero_division=0,
        ),
    }
    with open(args.output_dir / f"{name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote PyTorch evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
