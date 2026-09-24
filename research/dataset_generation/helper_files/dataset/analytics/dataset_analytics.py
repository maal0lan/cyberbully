#!/usr/bin/env python3
"""
Cyberbullying Dataset Analytics & Visualization Suite
=====================================================
Generates publication-quality data science visualizations and statistical reports
for the merged cyberbullying dataset.

Visualizations generated:
1. 01_label_distribution.png       - Class balance (0 vs 1) and source breakdown
2. 02_category_distribution.png    - Full 20-category distribution by class
3. 03_word_frequency_unigrams.png  - Top 25 unigrams (Bullying vs Non-Bullying)
4. 04_word_frequency_bigrams.png   - Top 20 bigrams (Bullying vs Non-Bullying)
5. 05_text_length_and_tokens.png   - Character length, word count & boxplots
6. 06_augmentation_analysis.png    - Adversarial noise & perturbation distribution
7. 07_target_type_distribution.png - Targeted entity distribution (individual, group, etc.)

Also exports:
- summary_statistics.json          - Machine-readable dataset profile & metrics
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# Visual Styling Config
PALETTE_CLASSES = {0: "#2a9d8f", 1: "#e76f51"}
PALETTE_SOURCES = {"explicit": "#457b9d", "non_explicit": "#e9c46a"}
COLOR_PRIMARY = "#264653"
COLOR_ACCENT = "#e76f51"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def get_default_paths():
    current_dir = Path(__file__).resolve().parent
    dataset_file = current_dir.parent / "cyberbullying_merged_dataset.csv"
    plots_dir = current_dir / "plots"
    stats_file = current_dir / "summary_statistics.json"
    return dataset_file, plots_dir, stats_file


def clean_text_for_vocab(text: str) -> str:
    """Lightweight regex cleaning for vocabulary frequency analysis."""
    if not isinstance(text, str):
        return ""
    # Normalize URLs, mentions, and symbols
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def plot_label_distribution(df: pd.DataFrame, output_path: Path):
    """Plot 1: Primary label distribution and source cross-tabulation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Donut chart for gen_label
    label_counts = df["gen_label"].value_counts().sort_index()
    labels = ["Not Cyberbullying (0)", "Cyberbullying (1)"]
    colors = [PALETTE_CLASSES[0], PALETTE_CLASSES[1]]
    total = len(df)

    wedges, texts, autotexts = axes[0].pie(
        label_counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct*total/100)):,} rows)",
        startangle=140,
        colors=colors,
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
    axes[0].set_title("Overall Binary Class Balance (gen_label)", fontsize=14, fontweight="bold", pad=15)

    # Right: Grouped bar chart by dataset_source
    ct = pd.crosstab(df["dataset_source"], df["gen_label"])
    ct.columns = ["Not Bullying (0)", "Bullying (1)"]
    x = np.arange(len(ct.index))
    width = 0.35

    rects1 = axes[1].bar(x - width/2, ct["Not Bullying (0)"], width, label="Not Bullying (0)", color=PALETTE_CLASSES[0], edgecolor="none", alpha=0.9)
    rects2 = axes[1].bar(x + width/2, ct["Bullying (1)"], width, label="Bullying (1)", color=PALETTE_CLASSES[1], edgecolor="none", alpha=0.9)

    axes[1].set_title("Class Breakdown by Dataset Source", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace("_", " ").title() for s in ct.index], fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=12)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Add count labels on top of bars
    for rects in [rects1, rects2]:
        for bar in rects:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_path.name}")


def plot_category_distribution(df: pd.DataFrame, output_path: Path):
    """Plot 2: Detailed category breakdown colored by cyberbullying label."""
    # Group by category and label
    cat_summary = df.groupby(["category", "gen_label"]).size().unstack(fill_value=0)
    cat_summary["total"] = cat_summary.sum(axis=1)
    cat_summary = cat_summary.sort_values(by="total", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    y = np.arange(len(cat_summary))
    height = 0.65

    # Stacked horizontal bars
    p1 = ax.barh(y, cat_summary.get(0, 0), height, label="Not Cyberbullying (0)", color=PALETTE_CLASSES[0], alpha=0.9)
    p2 = ax.barh(y, cat_summary.get(1, 0), height, left=cat_summary.get(0, 0), label="Cyberbullying (1)", color=PALETTE_CLASSES[1], alpha=0.9)

    ax.set_yticks(y)
    clean_labels = [c.replace("_", " ").capitalize() for c in cat_summary.index]
    ax.set_yticklabels(clean_labels, fontsize=11)
    ax.set_xlabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("Distribution Across All 20 Semantic Categories", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=11)

    # Annotate total on bar ends
    for i, total_val in enumerate(cat_summary["total"]):
        ax.annotate(
            f" {total_val:,}",
            xy=(total_val, i),
            va="center", ha="left",
            fontsize=9.5, fontweight="bold", color="#333333"
        )

    ax.set_xlim(0, max(cat_summary["total"]) * 1.12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_path.name}")


def plot_word_frequencies(df: pd.DataFrame, output_unigram_path: Path, output_bigram_path: Path):
    """Plots 3 & 4: Unigram and Bigram frequencies comparing Bullying vs Non-Bullying."""
    clean_texts_bully = df[df["gen_label"] == 1]["text"].dropna().apply(clean_text_for_vocab)
    clean_texts_benign = df[df["gen_label"] == 0]["text"].dropna().apply(clean_text_for_vocab)

    # Custom stop words filter
    stopwords = list(CountVectorizer(stop_words="english").get_stop_words()) + [
        "like", "just", "don", "ve", "ll", "did", "m", "re", "s", "t", "can", "url", "user"
    ]

    # --- 1. UNIGRAMS ---
    vec_uni = CountVectorizer(stop_words=stopwords, max_features=25, ngram_range=(1, 1))

    # Bullying Unigrams
    dtm_b = vec_uni.fit_transform(clean_texts_bully)
    uni_b = pd.DataFrame({"word": vec_uni.get_feature_names_out(), "count": np.asarray(dtm_b.sum(axis=0)).flatten()})
    uni_b = uni_b.sort_values(by="count", ascending=True)

    # Benign Unigrams
    dtm_nb = vec_uni.fit_transform(clean_texts_benign)
    uni_nb = pd.DataFrame({"word": vec_uni.get_feature_names_out(), "count": np.asarray(dtm_nb.sum(axis=0)).flatten()})
    uni_nb = uni_nb.sort_values(by="count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].barh(uni_nb["word"], uni_nb["count"], color=PALETTE_CLASSES[0], alpha=0.9, height=0.7)
    axes[0].set_title("Top 25 Words: Non-Cyberbullying (0)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Frequency Count", fontsize=11)
    for i, v in enumerate(uni_nb["count"]):
        axes[0].text(v + 15, i, f" {v:,}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, max(uni_nb["count"]) * 1.15)

    axes[1].barh(uni_b["word"], uni_b["count"], color=PALETTE_CLASSES[1], alpha=0.9, height=0.7)
    axes[1].set_title("Top 25 Words: Cyberbullying (1)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Frequency Count", fontsize=11)
    for i, v in enumerate(uni_b["count"]):
        axes[1].text(v + 15, i, f" {v:,}", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xlim(0, max(uni_b["count"]) * 1.15)

    plt.suptitle("Comparative Vocabulary Frequency Analysis (Unigrams)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_unigram_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_unigram_path.name}")

    # --- 2. BIGRAMS ---
    vec_bi = CountVectorizer(stop_words=stopwords, max_features=20, ngram_range=(2, 2))

    dtm_b_bi = vec_bi.fit_transform(clean_texts_bully)
    bi_b = pd.DataFrame({"ngram": vec_bi.get_feature_names_out(), "count": np.asarray(dtm_b_bi.sum(axis=0)).flatten()})
    bi_b = bi_b.sort_values(by="count", ascending=True)

    dtm_nb_bi = vec_bi.fit_transform(clean_texts_benign)
    bi_nb = pd.DataFrame({"ngram": vec_bi.get_feature_names_out(), "count": np.asarray(dtm_nb_bi.sum(axis=0)).flatten()})
    bi_nb = bi_nb.sort_values(by="count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].barh(bi_nb["ngram"], bi_nb["count"], color=PALETTE_CLASSES[0], alpha=0.9, height=0.7)
    axes[0].set_title("Top 20 Bigrams: Non-Cyberbullying (0)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Frequency Count", fontsize=11)
    for i, v in enumerate(bi_nb["count"]):
        axes[0].text(v + 5, i, f" {v:,}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, max(bi_nb["count"]) * 1.15)

    axes[1].barh(bi_b["ngram"], bi_b["count"], color=PALETTE_CLASSES[1], alpha=0.9, height=0.7)
    axes[1].set_title("Top 20 Bigrams: Cyberbullying (1)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Frequency Count", fontsize=11)
    for i, v in enumerate(bi_b["count"]):
        axes[1].text(v + 5, i, f" {v:,}", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xlim(0, max(bi_b["count"]) * 1.15)

    plt.suptitle("Semantic Context & Phrase Patterns (Bigrams)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_bigram_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_bigram_path.name}")


def plot_text_lengths(df: pd.DataFrame, output_path: Path):
    """Plot 5: Character length, word count distributions and category boxplots."""
    df_plot = df.copy()
    df_plot["char_length"] = df_plot["text"].astype(str).str.len()
    df_plot["word_count"] = df_plot["text"].astype(str).str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Top Left: Character Length KDE
    sns.kdeplot(
        data=df_plot,
        x="char_length",
        hue="gen_label",
        palette=PALETTE_CLASSES,
        fill=True,
        common_norm=False,
        alpha=0.35,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Character Length Distribution (KDE Density)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Character Count")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_xlim(0, df_plot["char_length"].quantile(0.99))
    axes[0, 0].legend(labels=["Cyberbullying (1)", "Not Cyberbullying (0)"], loc="upper right")

    # Top Right: Word Count KDE
    sns.kdeplot(
        data=df_plot,
        x="word_count",
        hue="gen_label",
        palette=PALETTE_CLASSES,
        fill=True,
        common_norm=False,
        alpha=0.35,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("Word Count Distribution (KDE Density)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Word Count")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_xlim(0, df_plot["word_count"].quantile(0.99))
    axes[0, 1].legend(labels=["Cyberbullying (1)", "Not Cyberbullying (0)"], loc="upper right")

    # Bottom Left: Word Count Boxplot by Label
    axes[1, 0].set_xticks([0, 1])
    sns.boxplot(
        data=df_plot,
        x="gen_label",
        y="word_count",
        hue="gen_label",
        palette=[PALETTE_CLASSES[0], PALETTE_CLASSES[1]],
        legend=False,
        ax=axes[1, 0],
        width=0.4,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
    )
    axes[1, 0].set_title("Word Count Distribution Comparison", fontsize=13, fontweight="bold")
    axes[1, 0].set_xticklabels(["Not Cyberbullying (0)", "Cyberbullying (1)"], fontweight="bold")
    axes[1, 0].set_ylabel("Word Count")
    axes[1, 0].set_ylim(0, df_plot["word_count"].quantile(0.995))

    # Bottom Right: Avg Word Count per Category (Top 12)
    top_cats = df_plot.groupby("category")["word_count"].mean().sort_values(ascending=False).head(12)
    axes[1, 1].barh(
        [c.replace("_", " ").capitalize() for c in top_cats.index][::-1],
        top_cats.values[::-1],
        color=COLOR_PRIMARY,
        alpha=0.85,
        height=0.65
    )
    axes[1, 1].set_title("Mean Word Count Across Top Categories", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Average Words per Message")
    for i, v in enumerate(top_cats.values[::-1]):
        axes[1, 1].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9.5, fontweight="bold")
    axes[1, 1].set_xlim(0, max(top_cats.values) * 1.15)

    plt.suptitle("Text Length & Tokenization Characteristics", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_path.name}")


def plot_augmentation_analysis(df: pd.DataFrame, output_path: Path):
    """Plot 6: Synthetic noise and adversarial perturbation breakdown."""
    df_aug = df[df["is_augmented"] == 1]
    if len(df_aug) == 0:
        print("   [!] No augmented rows found to plot.")
        return

    aug_counts = df_aug["augmentation_type"].value_counts().head(12)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Augmentation Types Bar
    y = np.arange(len(aug_counts))
    axes[0].barh(y, aug_counts.values, color="#457b9d", alpha=0.85, height=0.65)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([a.replace("_", " ").title() for a in aug_counts.index], fontsize=10)
    axes[0].set_xlabel("Number of Augmented Samples", fontsize=11)
    axes[0].set_title("Top Synthetic Noise / Perturbation Types", fontsize=13, fontweight="bold")
    for i, v in enumerate(aug_counts.values):
        axes[0].text(v + 2, i, f" {v:,}", va="center", fontsize=9.5, fontweight="bold")
    axes[0].set_xlim(0, max(aug_counts.values) * 1.15)
    axes[0].invert_yaxis()

    # Right: Augmentation by Class
    ct = pd.crosstab(df_aug["augmentation_type"].apply(lambda x: x.split("+")[0]), df_aug["gen_label"])
    ct.columns = ["Not Bullying (0)", "Bullying (1)"]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).head(7).index]

    ct.plot(
        kind="bar",
        ax=axes[1],
        color=[PALETTE_CLASSES[0], PALETTE_CLASSES[1]],
        edgecolor="none",
        alpha=0.9,
        width=0.7
    )
    axes[1].set_title("Primary Noise Perturbations by Target Class", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Base Perturbation", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_xticklabels([x.replace("_", " ").title() for x in ct.index], rotation=30, ha="right")
    axes[1].legend(frameon=True, facecolor="white")

    plt.suptitle("Adversarial Robustness & Augmentation Distribution", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_path.name}")


def plot_target_types(df: pd.DataFrame, output_path: Path):
    """Plot 7: Target type distribution (individual vs group vs self vs topic)."""
    # Clean and standardize top target types
    targets = df["target_type"].astype(str).str.strip().str.lower()
    top_targets = targets.value_counts().head(8)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        [t.capitalize() for t in top_targets.index],
        top_targets.values,
        color=COLOR_PRIMARY,
        alpha=0.85,
        width=0.55
    )

    ax.set_title("Distribution of Targeted Entities in Text", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Target Type", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}\n({height/len(df)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold"
        )

    ax.set_ylim(0, max(top_targets.values) * 1.15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {output_path.name}")


def compute_and_save_summary(df: pd.DataFrame, output_path: Path):
    """Computes and dumps quantitative profile into JSON format."""
    df_work = df.copy()
    df_work["char_len"] = df_work["text"].astype(str).str.len()
    df_work["word_cnt"] = df_work["text"].astype(str).str.split().str.len()

    stats = {
        "dataset_overview": {
            "total_rows": int(len(df_work)),
            "total_columns": int(len(df_work.columns)),
            "column_names": list(df_work.columns),
            "memory_usage_mb": round(df_work.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        },
        "class_balance_gen_label": {
            "cyberbullying_1": int((df_work["gen_label"] == 1).sum()),
            "not_cyberbullying_0": int((df_work["gen_label"] == 0).sum()),
            "cyberbullying_pct": round(float((df_work["gen_label"] == 1).mean() * 100), 2),
            "not_cyberbullying_pct": round(float((df_work["gen_label"] == 0).mean() * 100), 2),
            "recommended_class_weights": {
                "0": round(float(len(df_work) / (2.0 * (df_work["gen_label"] == 0).sum())), 4),
                "1": round(float(len(df_work) / (2.0 * (df_work["gen_label"] == 1).sum())), 4),
            }
        },
        "source_distribution": {
            k: int(v) for k, v in df_work["dataset_source"].value_counts().items()
        },
        "augmentation_summary": {
            "original_rows": int((df_work["is_augmented"] == 0).sum()),
            "augmented_rows": int((df_work["is_augmented"] == 1).sum()),
            "augmentation_percentage": round(float((df_work["is_augmented"] == 1).mean() * 100), 2),
            "augmentation_types": {
                k: int(v) for k, v in df_work[df_work["is_augmented"] == 1]["augmentation_type"].value_counts().head(10).items()
            }
        },
        "text_statistics": {
            "char_length": {
                "mean": round(float(df_work["char_len"].mean()), 1),
                "median": float(df_work["char_len"].median()),
                "min": int(df_work["char_len"].min()),
                "max": int(df_work["char_len"].max()),
                "p95": float(df_work["char_len"].quantile(0.95)),
            },
            "word_count": {
                "mean": round(float(df_work["word_cnt"].mean()), 1),
                "median": float(df_work["word_cnt"].median()),
                "min": int(df_work["word_cnt"].min()),
                "max": int(df_work["word_cnt"].max()),
                "p95": float(df_work["word_cnt"].quantile(0.95)),
            }
        },
        "category_counts": {
            k: int(v) for k, v in df_work["category"].value_counts().items()
        },
        "target_type_counts": {
            k: int(v) for k, v in df_work["target_type"].astype(str).str.lower().value_counts().head(10).items()
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"   [+] Saved summary statistics: {output_path.name}")
    return stats


def generate_all_analytics(dataset_path: Path, plots_dir: Path, stats_file: Path):
    print("=" * 65)
    print("   Cyberbullying Dataset Analytics & Visualization Suite")
    print("=" * 65)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding="utf-8", low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.\n")

    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations:")
    plot_label_distribution(df, plots_dir / "01_label_distribution.png")
    plot_category_distribution(df, plots_dir / "02_category_distribution.png")
    plot_word_frequencies(
        df,
        plots_dir / "03_word_frequency_unigrams.png",
        plots_dir / "04_word_frequency_bigrams.png"
    )
    plot_text_lengths(df, plots_dir / "05_text_length_and_tokens.png")
    plot_augmentation_analysis(df, plots_dir / "06_augmentation_analysis.png")
    plot_target_types(df, plots_dir / "07_target_type_distribution.png")

    print("\nCalculating summary metrics:")
    compute_and_save_summary(df, stats_file)

    print("\n" + "=" * 65)
    print("All analytics and plots generated successfully!")
    print(f"Plots directory: {plots_dir}")
    print(f"JSON metrics:    {stats_file}")
    print("=" * 65 + "\n")


def main():
    default_data, default_plots, default_stats = get_default_paths()

    parser = argparse.ArgumentParser(description="Generate visualizations and stats for cyberbullying dataset.")
    parser.add_argument("--dataset", type=Path, default=default_data, help=f"Path to CSV (default: {default_data})")
    parser.add_argument("--plots-dir", type=Path, default=default_plots, help=f"Directory to save plots (default: {default_plots})")
    parser.add_argument("--stats-file", type=Path, default=default_stats, help=f"Path to summary JSON (default: {default_stats})")

    args = parser.parse_args()
    generate_all_analytics(
        dataset_path=args.dataset,
        plots_dir=args.plots_dir,
        stats_file=args.stats_file
    )


if __name__ == "__main__":
    main()
