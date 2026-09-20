import os
import sys
import argparse
import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Translate dataset from English to French using Helsinki-NLP/opus-mt-tc-big-en-fr")
    parser.add_argument("--input", type=str, default="../dataset/checkpoint_generated.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="checkpoint_generated_french.csv", help="Path to output translated CSV file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save progress every N samples")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for generation")
    args = parser.parse_args()

    # Determine paths relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, args.input))
    output_path = os.path.normpath(os.path.join(script_dir, args.output))

    print(f"Loading input dataset from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Total rows in dataset: {len(df)}")

    # Check for existing checkpoint to support resuming
    start_idx = 0
    translated_texts = []
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if "text_fr" in existing_df.columns and len(existing_df) > 0:
                # Find how many rows were already translated
                valid_count = existing_df["text_fr"].notna().sum()
                if valid_count > 0 and valid_count <= len(df):
                    start_idx = int(valid_count)
                    translated_texts = existing_df["text_fr"].iloc[:start_idx].tolist()
                    print(f"Found existing checkpoint with {start_idx}/{len(df)} translated rows. Resuming from row {start_idx}...")
        except Exception as e:
            print(f"Could not read existing checkpoint: {e}. Starting from beginning.")
            translated_texts = []
            start_idx = 0

    # Model and device setup
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model & tokenizer: {model_name}...")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    texts_to_translate = df["text"].tolist()

    print(f"Starting translation from index {start_idx} to {len(df)} (batch size: {args.batch_size})...")
    
    batch_size = args.batch_size
    pbar = tqdm(total=len(df), initial=start_idx, desc="Translating en -> fr")

    try:
        for i in range(start_idx, len(df), batch_size):
            batch_texts = [str(t) if pd.notna(t) else "" for t in texts_to_translate[i:i + batch_size]]
            
            # Tokenize batch
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                translated_tokens = model.generate(**inputs, max_length=args.max_length)

            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(decoded)

            pbar.update(len(batch_texts))

            # Periodic checkpoint saving
            if len(translated_texts) % args.checkpoint_interval < batch_size or (i + batch_size >= len(df)):
                current_df = df.iloc[:len(translated_texts)].copy()
                current_df["text_fr"] = translated_texts
                # Also include translated text as primary 'text' in french dataset or keep side-by-side
                current_df.to_csv(output_path, index=False)
                
    except KeyboardInterrupt:
        print("\nTranslation paused by user. Saving current progress...")
        current_df = df.iloc[:len(translated_texts)].copy()
        current_df["text_fr"] = translated_texts
        current_df.to_csv(output_path, index=False)
        print(f"Saved {len(translated_texts)} rows to {output_path}")
        pbar.close()
        return
    except Exception as e:
        print(f"\nError during translation: {e}. Saving current progress...")
        current_df = df.iloc[:len(translated_texts)].copy()
        current_df["text_fr"] = translated_texts
        current_df.to_csv(output_path, index=False)
        print(f"Saved {len(translated_texts)} rows to {output_path}")
        pbar.close()
        raise e

    pbar.close()

    # Final save
    df["text_fr"] = translated_texts
    # We also create a clean French version with columns matching original schema: text (french), gen_label, category, target_type, source_word
    df_fr_clean = df.copy()
    df_fr_clean["text_en"] = df_fr_clean["text"]
    df_fr_clean["text"] = df_fr_clean["text_fr"]
    
    # Save both side-by-side and clean format
    df.to_csv(output_path, index=False)
    clean_output_path = os.path.join(script_dir, "checkpoint_generated_fr.csv")
    df_fr_clean.to_csv(clean_output_path, index=False)

    print(f"\n[DONE] Translation completed successfully!")
    print(f"Saved full comparison dataset with 'text_fr' to: {output_path}")
    print(f"Saved French dataset to: {clean_output_path}")

if __name__ == "__main__":
    main()
