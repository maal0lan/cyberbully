"""Compare the saved Keras and v0.1 DistilBERT models on the same test rows.

Run from the repository root:
    python old_vs_new_eval/compare_models.py

The saved DistilBERT test predictions are checked against test.csv before use.
The Keras model receives the same test.csv raw text through its built-in
TextVectorization layer.
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
    and os.environ.get("CYBERBULLY_COMPARE_UTF8_REEXEC") != "1"
):
    environment = os.environ.copy()
    environment["CYBERBULLY_COMPARE_UTF8_REEXEC"] = "1"
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERAS_MODEL = ROOT / "cyberbully_output" / "cyberbullying_detector_full.keras"
DEFAULT_RUN_DIR = ROOT / "cyberbully_v0.1_run"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_STYLES = {
    "Old Keras": {"linestyle": "--", "marker": "o"},
    "New DistilBERT v0.1": {"linestyle": "-", "marker": "s"},
}
METRIC_COLORS = {
    "accuracy": "#1f77b4",
    "precision": "#ff7f0e",
    "recall": "#2ca02c",
    "f1": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an old Keras vs new DistilBERT evaluation on shared test data."
    )
    parser.add_argument("--keras-model", type=Path, default=DEFAULT_KERAS_MODEL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def get_threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    positive_suffix_sum = np.cumsum(sorted_labels[::-1])[::-1]
    first_positive_index = np.searchsorted(sorted_scores, thresholds, side="left")
    predicted_positive_count = len(labels) - first_positive_index

    true_positives = np.zeros(len(thresholds), dtype=np.int64)
    valid_indices = first_positive_index < len(labels)
    true_positives[valid_indices] = positive_suffix_sum[
        first_positive_index[valid_indices]
    ]
    false_positives = predicted_positive_count - true_positives
    true_negatives = int(np.sum(labels == 0)) - false_positives
    false_negatives = int(np.sum(labels == 1)) - true_positives

    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros(len(thresholds), dtype=float),
        where=(true_positives + false_positives) != 0,
    )
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros(len(thresholds), dtype=float),
        where=(true_positives + false_negatives) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(thresholds), dtype=float),
        where=(precision + recall) != 0,
    )
    accuracy = (true_positives + true_negatives) / len(labels)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float):
    predictions = (scores >= threshold).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_cyberbullying": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_cyberbullying": float(recall_score(labels, predictions, zero_division=0)),
        "f1_cyberbullying": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "false_positive_count": int(np.sum((labels == 0) & (predictions == 1))),
        "false_negative_count": int(np.sum((labels == 1) & (predictions == 0))),
    }


def save_threshold_plot(
    labels: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    all_thresholds = np.unique(
        np.concatenate(
            [np.asarray([threshold]), *(scores for scores in scores_by_model.values())]
        )
    )
    metric_curves = {
        name: get_threshold_metrics(labels, scores, all_thresholds)
        for name, scores in scores_by_model.items()
    }
    threshold_values = {
        name: evaluate_at_threshold(labels, scores, threshold)
        for name, scores in scores_by_model.items()
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    threshold_metric_keys = {
        "accuracy": "accuracy",
        "precision": "precision_cyberbullying",
        "recall": "recall_cyberbullying",
        "f1": "f1_cyberbullying",
    }
    for model_name, scores in scores_by_model.items():
        style = MODEL_STYLES[model_name]
        for metric_name, metric_label in metric_labels.items():
            value = threshold_values[model_name][threshold_metric_keys[metric_name]]
            ax.plot(
                all_thresholds,
                metric_curves[model_name][metric_name],
                color=METRIC_COLORS[metric_name],
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{model_name} | {metric_label} = {value:.3f} @ {threshold:g}",
            )
            ax.scatter(
                [threshold],
                [value],
                color=METRIC_COLORS[metric_name],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.8,
                s=58,
                zorder=3,
            )

    ax.axvline(
        threshold,
        color="#555555",
        linestyle=":",
        linewidth=1.5,
        label=f"Selected threshold ({threshold:g})",
    )
    ax.set(
        title="Old Keras vs New DistilBERT: Metrics by Threshold",
        xlabel="Probability threshold",
        ylabel="Metric score",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8,
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return threshold_values


def save_roc_plot(
    labels: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
) -> dict[str, float]:
    aucs = {}
    fig, ax = plt.subplots(figsize=(8, 8))
    model_colors = {"Old Keras": "#e66101", "New DistilBERT v0.1": "#1f78b4"}
    for model_name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        aucs[model_name] = auc
        threshold_metrics = evaluate_at_threshold(labels, scores, threshold)
        true_positive_rate = threshold_metrics["recall_cyberbullying"]
        false_positive_rate = threshold_metrics["false_positive_count"] / int(
            np.sum(labels == 0)
        )
        ax.plot(
            fpr,
            tpr,
            color=model_colors[model_name],
            linewidth=2,
            linestyle=MODEL_STYLES[model_name]["linestyle"],
            label=f"{model_name} (ROC-AUC = {auc:.3f})",
        )
        ax.scatter(
            [false_positive_rate],
            [true_positive_rate],
            color=model_colors[model_name],
            marker=MODEL_STYLES[model_name]["marker"],
            edgecolor="white",
            linewidth=0.8,
            s=65,
            zorder=3,
            label=f"{model_name} @ threshold {threshold:g}",
        )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", label="Random classifier")
    ax.set(
        title="Old Keras vs New DistilBERT: ROC Curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return aucs


def save_pr_plot(
    labels: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
) -> dict[str, float]:
    average_precisions = {}
    fig, ax = plt.subplots(figsize=(8, 8))
    model_colors = {"Old Keras": "#8c510a", "New DistilBERT v0.1": "#018571"}
    for model_name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = float(average_precision_score(labels, scores))
        average_precisions[model_name] = average_precision
        threshold_metrics = evaluate_at_threshold(labels, scores, threshold)
        ax.plot(
            recall,
            precision,
            color=model_colors[model_name],
            linewidth=2,
            linestyle=MODEL_STYLES[model_name]["linestyle"],
            label=f"{model_name} (AP = {average_precision:.3f})",
        )
        ax.scatter(
            [threshold_metrics["recall_cyberbullying"]],
            [threshold_metrics["precision_cyberbullying"]],
            color=model_colors[model_name],
            marker=MODEL_STYLES[model_name]["marker"],
            edgecolor="white",
            linewidth=0.8,
            s=65,
            zorder=3,
            label=f"{model_name} @ threshold {threshold:g}",
        )
    ax.axhline(
        np.mean(labels),
        color="#777777",
        linestyle=":",
        label=f"Positive rate = {np.mean(labels):.3f}",
    )
    ax.set(
        title="Old Keras vs New DistilBERT: Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return average_precisions


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than 0.")

    test_path = args.run_dir / "test.csv"
    new_predictions_path = args.run_dir / "test_predictions.csv"
    for path in (args.keras_model, test_path, new_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file not found: {path}")

    test_data = pd.read_csv(test_path)
    new_predictions = pd.read_csv(new_predictions_path)
    for name, frame in (("test.csv", test_data), ("test_predictions.csv", new_predictions)):
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if "prob_bully" not in new_predictions.columns:
        raise ValueError("test_predictions.csv is missing the prob_bully score column.")
    if len(test_data) != len(new_predictions):
        raise ValueError("test.csv and test_predictions.csv have different row counts.")

    labels = pd.to_numeric(test_data["label"], errors="raise").to_numpy()
    prediction_labels = pd.to_numeric(
        new_predictions["label"], errors="raise"
    ).to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isin(prediction_labels, [0, 1]).all():
        raise ValueError("Test labels must contain only 0 and 1.")
    if not np.array_equal(labels, prediction_labels):
        raise ValueError("Saved DistilBERT predictions do not match test.csv labels/order.")
    test_texts = test_data["text"].fillna("").astype(str).to_numpy()
    prediction_texts = new_predictions["text"].fillna("").astype(str).to_numpy()
    if not np.array_equal(test_texts, prediction_texts):
        raise ValueError("Saved DistilBERT predictions do not match test.csv text/order.")
    labels = labels.astype(np.int32)
    if np.unique(labels).size != 2:
        raise ValueError("Shared test data must contain examples from both classes.")

    new_scores = pd.to_numeric(
        new_predictions["prob_bully"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(new_scores).all() or (new_scores < 0).any() or (new_scores > 1).any():
        raise ValueError("Saved DistilBERT probabilities must be finite values in [0, 1].")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load the older Keras model. "
            "Install it with: pip install tensorflow"
        ) from exc

    keras_model = load_model(args.keras_model, compile=False)
    keras_texts = test_texts.reshape(-1, 1)
    old_score_batches = []
    for start in range(0, len(keras_texts), args.batch_size):
        batch = tf.convert_to_tensor(
            keras_texts[start : start + args.batch_size], dtype=tf.string
        )
        old_score_batches.append(keras_model(batch, training=False).numpy())
    old_scores = np.concatenate(old_score_batches, axis=0).reshape(-1).astype(float)
    if len(old_scores) != len(labels):
        raise ValueError("Keras model did not return exactly one score per test row.")
    if not np.isfinite(old_scores).all() or (old_scores < 0).any() or (old_scores > 1).any():
        raise ValueError("Keras model scores must be finite probabilities in [0, 1].")

    scores_by_model = {
        "Old Keras": old_scores,
        "New DistilBERT v0.1": new_scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_plot = args.output_dir / "old_vs_new_threshold_metrics.png"
    roc_plot = args.output_dir / "old_vs_new_roc_curve.png"
    pr_plot = args.output_dir / "old_vs_new_precision_recall_curve.png"

    threshold_metrics = save_threshold_plot(
        labels, scores_by_model, args.threshold, threshold_plot
    )
    roc_aucs = save_roc_plot(labels, scores_by_model, args.threshold, roc_plot)
    average_precisions = save_pr_plot(
        labels, scores_by_model, args.threshold, pr_plot
    )
    results = {
        "dataset": str(test_path.resolve()),
        "rows": int(len(labels)),
        "positive_label": 1,
        "threshold": args.threshold,
        "models": {},
        "plots": {
            "threshold_metrics": str(threshold_plot.resolve()),
            "roc_curve": str(roc_plot.resolve()),
            "precision_recall_curve": str(pr_plot.resolve()),
        },
    }
    for model_name, scores in scores_by_model.items():
        results["models"][model_name] = {
            **threshold_metrics[model_name],
            "roc_auc": roc_aucs[model_name],
            "average_precision": average_precisions[model_name],
        }
    metrics_path = args.output_dir / "old_vs_new_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
