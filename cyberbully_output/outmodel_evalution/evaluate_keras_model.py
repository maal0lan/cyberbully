"""Evaluate cyberbullying_detector_full.keras on heldout_test_data.csv.

Run from the repository root with:
    python cyberbully_output/evaluate_keras_model.py

The model contains its own text vectorization layer, so this script passes the
CSV's raw ``text`` column directly to the model.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if (
    os.name == "nt"
    and not sys.flags.utf8_mode
    and os.environ.get("CYBERBULLY_EVAL_UTF8_REEXEC") != "1"
):
    environment = os.environ.copy()
    environment["CYBERBULLY_EVAL_UTF8_REEXEC"] = "1"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            env=environment,
        )
    )

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "cyberbullying_detector_full.keras"
DEFAULT_DATA = SCRIPT_DIR / "heldout_test_data.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved Keras cyberbullying model on held-out CSV data."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicting cyberbullying (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of texts to predict at a time (default: 128).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than 0.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Keras model not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {args.data}")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load this model. Install it with: pip install tensorflow"
        ) from exc

    data = pd.read_csv(args.data)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing_columns))}"
        )
    if data.empty:
        raise ValueError("The evaluation CSV contains no rows.")

    labels = pd.to_numeric(data["label"], errors="raise").to_numpy()
    if pd.isna(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("The CSV label column must contain only 0 and 1.")
    labels = labels.astype(np.int32)
    if np.unique(labels).size != 2:
        raise ValueError("The evaluation CSV must contain examples from both classes.")

    texts = np.asarray(data["text"].fillna("").astype(str), dtype=str).reshape(-1, 1)
    model = load_model(args.model, compile=False)
    score_batches = []
    for start in range(0, len(texts), args.batch_size):
        batch = tf.convert_to_tensor(
            texts[start : start + args.batch_size], dtype=tf.string
        )
        score_batches.append(model(batch, training=False).numpy())
    raw_scores = np.concatenate(score_batches, axis=0)
    if raw_scores.size != len(labels):
        raise ValueError(
            "Expected one prediction score per CSV row, but the model returned "
            f"shape {raw_scores.shape} for {len(labels)} rows."
        )
    scores = raw_scores.reshape(-1).astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("The model returned non-finite prediction scores.")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError(
            "The model outputs are not probabilities in [0, 1]; "
            "cannot calculate ROC-AUC/PR-AUC as configured."
        )

    predictions = (scores >= args.threshold).astype(np.int32)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "threshold": args.threshold,
        "test_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_cyberbullying": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall_cyberbullying": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_cyberbullying": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "pr_auc_method": "average precision",
        "false_positive_count": fp,
        "false_negative_count": fn,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": [
            ["true_negative", "false_positive"],
            ["false_negative", "true_positive"],
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "keras_evaluation_metrics.json"
    confusion_matrix_path = args.output_dir / "keras_confusion_matrix.png"
    threshold_curves_path = args.output_dir / "keras_threshold_metrics.png"
    roc_curve_path = args.output_dir / "keras_roc_curve.png"
    pr_curve_path = args.output_dir / "keras_precision_recall_curve.png"

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not cyberbullying (0)", "Cyberbullying (1)"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(f"Confusion Matrix (threshold = {args.threshold:g})")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=160)
    plt.close()

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    curve_thresholds = np.unique(np.append(scores, args.threshold))
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_positive_labels = labels[sorted_indices]
    positive_suffix_sum = np.cumsum(sorted_positive_labels[::-1])[::-1]
    first_predicted_index = np.searchsorted(
        sorted_scores, curve_thresholds, side="left"
    )
    predicted_positive_count = len(labels) - first_predicted_index
    true_positives = np.zeros(len(curve_thresholds), dtype=np.int64)
    valid_prediction_indices = first_predicted_index < len(labels)
    true_positives[valid_prediction_indices] = positive_suffix_sum[
        first_predicted_index[valid_prediction_indices]
    ]
    false_positives = predicted_positive_count - true_positives
    true_negatives = negative_count - false_positives
    curve_recall = true_positives / positive_count
    curve_precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_positives) != 0,
    )
    curve_f1 = np.divide(
        2 * curve_precision * curve_recall,
        curve_precision + curve_recall,
        out=np.zeros_like(curve_precision, dtype=float),
        where=(curve_precision + curve_recall) != 0,
    )
    curve_accuracy = (true_positives + true_negatives) / len(labels)

    selected_precision = metrics["precision_cyberbullying"]
    selected_recall = metrics["recall_cyberbullying"]
    selected_f1 = metrics["f1_cyberbullying"]
    selected_values = {
        "Accuracy": metrics["accuracy"],
        "Precision": selected_precision,
        "Recall": selected_recall,
        "F1": selected_f1,
    }
    colors = {
        "Accuracy": "#1f77b4",
        "Precision": "#ff7f0e",
        "Recall": "#2ca02c",
        "F1": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in (
        ("Accuracy", curve_accuracy),
        ("Precision", curve_precision),
        ("Recall", curve_recall),
        ("F1", curve_f1),
    ):
        color = colors[name]
        value_at_threshold = selected_values[name]
        ax.plot(
            curve_thresholds,
            values,
            color=color,
            linewidth=2,
            label=f"{name} ({value_at_threshold:.3f} at {args.threshold:g})",
        )
        ax.scatter(
            [args.threshold],
            [value_at_threshold],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=55,
            zorder=3,
        )
    ax.axvline(
        args.threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Selected threshold ({args.threshold:g})",
    )
    ax.set(
        title="Classification Metrics by Decision Threshold",
        xlabel="Probability threshold",
        ylabel="Metric score",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(threshold_curves_path, dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#1f77b4",
        linewidth=2,
        label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Random classifier")
    ax.set(
        title="Receiver Operating Characteristic",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_curve_path, dpi=160)
    plt.close(fig)

    precision_values, recall_values, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        recall_values,
        precision_values,
        color="#9467bd",
        linewidth=2,
        label=f"Precision-recall curve (AP = {metrics['pr_auc']:.3f})",
    )
    ax.set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(pr_curve_path, dpi=160)
    plt.close(fig)

    metrics["plots"] = {
        "confusion_matrix": str(confusion_matrix_path.resolve()),
        "threshold_metrics": str(threshold_curves_path.resolve()),
        "roc_curve": str(roc_curve_path.resolve()),
        "precision_recall_curve": str(pr_curve_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Threshold metrics plot saved to: {threshold_curves_path}")
    print(f"ROC curve saved to: {roc_curve_path}")
    print(f"Precision-recall curve saved to: {pr_curve_path}")


if __name__ == "__main__":
    main()
