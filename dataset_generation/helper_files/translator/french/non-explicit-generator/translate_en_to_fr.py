#!/usr/bin/env python3
"""
Translate Non-Explicit Cyberbullying Dataset from English to French
===================================================================
Translates:
  dataset_generation/helper_files/gen/non-explicit-generator/output_clean/generated_raw_clean.csv
To French and stores outputs in:
  dataset_generation/helper_files/translator/french/non-explicit-generator/output_clean/

Features:
- Fast GPU acceleration with PyTorch CUDA.
- Checkpointed and fully resumable if interrupted.
- Generates:
    1. generated_raw_clean_french.csv (text, gen_label, category, target_type, source_topic, text_fr)
    2. generated_raw_clean_fr.csv (text [French], gen_label, category, target_type, source_topic, text_fr, text_en)
    3. checkpoint_generated_french.csv & checkpoint_generated_fr.csv (matching explicit-generator naming)
"""

import os
import sys
import types
import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

# Ensure clean imports under Windows Smart App Control
sp_mock = types.ModuleType("sentencepiece")
sp_mock.SentencePieceProcessor = lambda: None
sp_mock.__spec__ = types.SimpleNamespace(origin="mock")
sys.modules["sentencepiece"] = sp_mock

sk_mock = types.ModuleType("sklearn")
sk_mock.__spec__ = types.SimpleNamespace(origin="mock")
sk_metrics = types.ModuleType("sklearn.metrics")
sk_metrics.roc_curve = lambda *a, **k: None
sk_mock.metrics = sk_metrics
sys.modules["sklearn"] = sk_mock
sys.modules["sklearn.metrics"] = sk_metrics

import transformers.utils.import_utils as iu
iu.is_sentencepiece_available = lambda: True

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_default_paths():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[4]  # c:/Users/Praveen/cyberbully
    input_file = project_root / "dataset_generation" / "helper_files" / "gen" / "non-explicit-generator" / "output_clean" / "generated_raw_clean.csv"
    output_dir = current_dir / "output_clean"
    return input_file, output_dir


def main():
    default_input, default_output_dir = get_default_paths()

    parser = argparse.ArgumentParser(description="Translate non-explicit cyberbullying dataset to French.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir, help="Directory to store translated files")
    parser.add_argument("--model", type=str, default="google-t5/t5-small", help="Pretrained Seq2Seq translation model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU translation")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save progress every N samples")
    parser.add_argument("--max_length", type=int, default=128, help="Max generation token length")
    args = parser.parse_args()

    print("=" * 65)
    print("   Non-Explicit Dataset English -> French Translator")
    print("=" * 65)

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_checkpoint = output_dir / "generated_raw_clean_french.csv"

    print(f"\n[1/4] Loading input dataset:")
    print(f"      Source: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"      Loaded {len(df):,} rows with columns: {list(df.columns)}")

    # Check for existing checkpoint to support resuming
    start_idx = 0
    translated_texts = []

    if output_checkpoint.is_file():
        try:
            existing_df = pd.read_csv(output_checkpoint, encoding="utf-8")
            if "text_fr" in existing_df.columns and len(existing_df) > 0:
                valid_count = existing_df["text_fr"].notna().sum()
                if 0 < valid_count <= len(df):
                    start_idx = int(valid_count)
                    translated_texts = existing_df["text_fr"].iloc[:start_idx].tolist()
                    print(f"      [RESUME] Found existing checkpoint with {start_idx}/{len(df)} translated rows.")
        except Exception as e:
            print(f"      [!] Could not read existing checkpoint ({e}). Starting fresh.")
            translated_texts = []
            start_idx = 0

    # Device & Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/4] Initializing model & tokenizer on: {device.upper()}")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    prefix = "translate English to French: "
    texts_to_translate = df["text"].tolist()

    print(f"\n[3/4] Translating from row {start_idx:,} to {len(df):,} (Batch Size: {args.batch_size})...")
    batch_size = args.batch_size
    pbar = tqdm(total=len(df), initial=start_idx, desc="Translating EN -> FR")

    try:
        for i in range(start_idx, len(df), batch_size):
            batch_texts = [str(t) if pd.notna(t) else "" for t in texts_to_translate[i:i + batch_size]]
            prefixed_batch = [prefix + t for t in batch_texts]

            inputs = tokenizer(
                prefixed_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=args.max_length)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_texts.extend(decoded)
            pbar.update(len(batch_texts))

            # Periodic checkpoint save
            if len(translated_texts) % args.checkpoint_interval < batch_size or (i + batch_size >= len(df)):
                current_df = df.iloc[:len(translated_texts)].copy()
                current_df["text_fr"] = translated_texts
                current_df.to_csv(output_checkpoint, index=False, encoding="utf-8")

    except KeyboardInterrupt:
        print("\n\n[!] Translation paused by user. Saving current checkpoint...")
        current_df = df.iloc[:len(translated_texts)].copy()
        current_df["text_fr"] = translated_texts
        current_df.to_csv(output_checkpoint, index=False, encoding="utf-8")
        print(f"Saved {len(translated_texts):,} rows to {output_checkpoint}")
        pbar.close()
        return

    except Exception as e:
        print(f"\n\n[!] Error during translation: {e}. Saving current checkpoint...")
        current_df = df.iloc[:len(translated_texts)].copy()
        current_df["text_fr"] = translated_texts
        current_df.to_csv(output_checkpoint, index=False, encoding="utf-8")
        pbar.close()
        raise e

    pbar.close()

    # Final Saves
    print(f"\n[4/4] Writing finalized French datasets...")
    df["text_fr"] = translated_texts

    # 1. generated_raw_clean_french.csv
    file_french = output_dir / "generated_raw_clean_french.csv"
    df.to_csv(file_french, index=False, encoding="utf-8")

    # 2. generated_raw_clean_fr.csv (text replaced with French, text_en added)
    df_clean_fr = df.copy()
    df_clean_fr["text_en"] = df_clean_fr["text"]
    df_clean_fr["text"] = df_clean_fr["text_fr"]
    file_fr = output_dir / "generated_raw_clean_fr.csv"
    df_clean_fr.to_csv(file_fr, index=False, encoding="utf-8")

    # 3. checkpoint_generated_french.csv & checkpoint_generated_fr.csv (mirroring explicit generator naming)
    df.to_csv(output_dir / "checkpoint_generated_french.csv", index=False, encoding="utf-8")
    df_clean_fr.to_csv(output_dir / "checkpoint_generated_fr.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("Translation completed successfully!")
    print(f"Total Rows Translated: {len(df):,}")
    print(f"Output Directory:      {output_dir}")
    print(f"Generated Files:")
    print(f"  1. {file_french.name:<30} ({file_french.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  2. {file_fr.name:<30} ({file_fr.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  3. checkpoint_generated_french.csv")
    print(f"  4. checkpoint_generated_fr.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
