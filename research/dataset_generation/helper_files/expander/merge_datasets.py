#!/usr/bin/env python3
"""
Merge Cyberbullying Datasets
============================
Merges explicit and non-explicit augmented datasets into a unified dataset:
- File 1: dataset_generation/helper_files/expander/explicit_dataset_generation/cyberbullying_augmented_noise.csv
- File 2: dataset_generation/helper_files/expander/non-explicit_dataset_generation/cyberbullying_augmented_noise.csv
- Output: dataset_generation/helper_files/dataset/cyberbullying_merged_dataset.csv

Adds a 'dataset_source' column ('explicit' / 'non_explicit') to track provenance.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd


def get_default_paths():
    current_dir = Path(__file__).resolve().parent
    explicit_file = current_dir / "explicit_dataset_generation" / "cyberbullying_augmented_noise.csv"
    non_explicit_file = current_dir / "non-explicit_dataset_generation" / "cyberbullying_augmented_noise.csv"
    output_dir = current_dir.parent / "dataset"
    output_file = output_dir / "cyberbullying_merged_dataset.csv"
    return explicit_file, non_explicit_file, output_file


def merge_datasets(explicit_path: Path, non_explicit_path: Path, output_path: Path, shuffle: bool = False, seed: int = 42) -> pd.DataFrame:
    print("=" * 60)
    print("   Cyberbullying Dataset Merger")
    print("=" * 60)

    # 1. Validation
    if not explicit_path.is_file():
        raise FileNotFoundError(f"Explicit dataset not found at: {explicit_path}")
    if not non_explicit_path.is_file():
        raise FileNotFoundError(f"Non-explicit dataset not found at: {non_explicit_path}")

    print(f"\n[1/4] Reading explicit dataset:     {explicit_path.name}")
    df_explicit = pd.read_csv(explicit_path, encoding="utf-8", low_memory=False)
    df_explicit["dataset_source"] = "explicit"
    print(f"      -> Loaded {len(df_explicit):,} rows, {len(df_explicit.columns)} columns")

    print(f"\n[2/4] Reading non-explicit dataset: {non_explicit_path.name}")
    df_non_explicit = pd.read_csv(non_explicit_path, encoding="utf-8", low_memory=False)
    df_non_explicit["dataset_source"] = "non_explicit"
    print(f"      -> Loaded {len(df_non_explicit):,} rows, {len(df_non_explicit.columns)} columns")

    # 2. Concatenate
    print(f"\n[3/4] Merging datasets...")
    df_merged = pd.concat([df_explicit, df_non_explicit], ignore_index=True)

    if shuffle:
        print(f"      -> Shuffling dataset with random_state={seed}")
        df_merged = df_merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 3. Save Merged Dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding="utf-8")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # 4. Save Augmented-Only Dataset
    # Filter by augmentation_id not null / is_augmented == 1
    is_aug_mask = df_merged["augmentation_id"].notna() | (df_merged["is_augmented"] == 1)
    df_augmented = df_merged[is_aug_mask].copy()
    
    # User-requested spelling
    aug_output_path = output_path.parent / "cuberbully_augumented_dataset.csv"
    df_augmented.to_csv(aug_output_path, index=False, encoding="utf-8")
    # Also save standard spelling alias for convenience
    aug_output_path_alias = output_path.parent / "cyberbully_augmented_dataset.csv"
    df_augmented.to_csv(aug_output_path_alias, index=False, encoding="utf-8")

    # 5. Save Raw-Only (Non-Augmented) Dataset
    df_raw = df_merged[~is_aug_mask].copy()
    raw_output_path = output_path.parent / "cyberbully_raw_dataset.csv"
    df_raw.to_csv(raw_output_path, index=False, encoding="utf-8")

    print(f"\n[4/4] Successfully saved datasets to {output_path.parent}:")
    print(f"      1. Full Merged:      {output_path.name:<35} -> {len(df_merged):,} rows ({file_size_mb:.2f} MB)")
    print(f"      2. Augmented Only:   {aug_output_path.name:<35} -> {len(df_augmented):,} rows ({aug_output_path.stat().st_size / 1024:.1f} KB)")
    print(f"      3. Raw Non-Augmented:{raw_output_path.name:<35} -> {len(df_raw):,} rows ({raw_output_path.stat().st_size / (1024*1024):.2f} MB)")

    # 6. Summary Statistics
    print("\n" + "-" * 60)
    print("DATASET PROFILE & DISTRIBUTION SUMMARY:")
    print("-" * 60)

    print("\n1. Distribution by Dataset Source:")
    src_dist = df_merged["dataset_source"].value_counts()
    for src, count in src_dist.items():
        print(f"   - {src:<15}: {count:6,} ({count / len(df_merged) * 100:.2f}%)")

    print("\n2. Distribution by Primary Label (gen_label):")
    label_map = {0: "Not Cyberbullying (0)", 1: "Cyberbullying (1)"}
    lbl_dist = df_merged["gen_label"].value_counts()
    for lbl, count in lbl_dist.items():
        name = label_map.get(lbl, str(lbl))
        print(f"   - {name:<25}: {count:6,} ({count / len(df_merged) * 100:.2f}%)")

    print("\n3. Cross-Tabulation (Source vs gen_label):")
    ct = pd.crosstab(df_merged["dataset_source"], df_merged["gen_label"], margins=True)
    ct.columns = ["Not Bullying (0)", "Bullying (1)", "Total"]
    print(ct.to_string())

    print("\n4. Augmentation Breakdown:")
    aug_dist = df_merged["is_augmented"].value_counts()
    print(f"   - Original samples : {aug_dist.get(0, 0):6,} ({aug_dist.get(0, 0) / len(df_merged) * 100:.2f}%)")
    print(f"   - Augmented samples: {aug_dist.get(1, 0):6,} ({aug_dist.get(1, 0) / len(df_merged) * 100:.2f}%)")

    print(f"\n5. Total Unique Categories: {df_merged['category'].nunique()}")
    for cat, count in df_merged["category"].value_counts().items():
        print(f"   - {cat:<36}: {count:5,} rows")

    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print("=" * 60 + "\n")

    return df_merged


def main():
    default_explicit, default_non_explicit, default_output = get_default_paths()

    parser = argparse.ArgumentParser(description="Merge explicit and non-explicit augmented cyberbullying datasets.")
    parser.add_argument(
        "--explicit",
        type=Path,
        default=default_explicit,
        help=f"Path to explicit CSV (default: {default_explicit})"
    )
    parser.add_argument(
        "--non-explicit",
        type=Path,
        default=default_non_explicit,
        help=f"Path to non-explicit CSV (default: {default_non_explicit})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to output merged CSV (default: {default_output})"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to randomly shuffle the merged rows"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed if shuffling (default: 42)"
    )

    args = parser.parse_args()
    merge_datasets(
        explicit_path=args.explicit,
        non_explicit_path=args.non_explicit,
        output_path=args.output,
        shuffle=args.shuffle,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
