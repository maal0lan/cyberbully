import csv
from pathlib import Path
from transformers import pipeline

# 1. Initialize the pipeline
# Passing the model name directly handles tokenizers and models automatically under the hood
generator = pipeline(
    'text-generation', 
    model='openai-community/gpt2', 
    device=0  # Set to 0 if you are running on a CUDA GPU
)

# 2. Define file paths (Adjust these to match your actual file names/locations)
input_txt_path = Path(r"C:\Users\Priya\Desktop\maalolan\epic_maalolan\contributions\cyberbully\cyberbully_\dataset\bad_words_all_languages\data_words\English") 
output_csv_path = Path(r"C:\Users\Priya\Desktop\maalolan\epic_maalolan\contributions\cyberbully\cyberbully_\dataset\bad_words_all_languages\data_sentence\English")

# 3. Read the 403 words from your file
if not input_txt_path.exists():
    raise FileNotFoundError(f"Could not find your input file at: {input_txt_path.resolve()}")

with open(input_txt_path, "r", encoding="utf-8") as f:
    # Read lines, strip whitespace, and filter out empty lines
    bad_words = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(bad_words)} words. Starting generation...")

# 4. Open CSV for writing and loop through the words
with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # Write headers for your DeBERTa fine-tuning dataset
    writer.writerow(["target_word", "generated_sentence"])
    
    for idx, word in enumerate(bad_words, 1):
        print(f"[{idx}/{len(bad_words)}] Generating sentences for: {word}")
        
        # Crafting a strict prompt to force GPT-2 to output just the statement/comment
        prompt = f"""
Generate a short toxic online comment.

Rules:
- Must be ONE sentence only
- Must sound like a real social media insult
- Must include the word: {word}
- Do NOT explain anything
- Do NOT repeat the prompt
- Keep it under 15 words

Examples:
you are such an idiot
shut up you loser
nobody likes you

Comment:
"""
        
        try:
            outputs = generator(
    prompt,
    max_new_tokens=30,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=50256
) # Explicitly set EOS token as padding to avoid warnings
            
            
            for out in outputs:
                # Remove the prompt prefix to get only the new generation
                generated_text = out['generated_text'].replace(prompt, "").strip()
                
                # GPT-2 might continue past the closing quotation mark or add extra lines.
                # We split by newline and grab the first sentence block for cleanliness.
                clean_sentence = generated_text.split("\n")[0].replace('"', '').strip()
                
                # Make sure it actually kept the word in the context
                if word.lower() in clean_sentence.lower() and len(clean_sentence) > 5:
                    writer.writerow([word, clean_sentence])
                    
        except Exception as e:
            print(f"Error generating for word '{word}': {e}")
            continue

print(f"Done! Dataset saved successfully to {output_csv_path.resolve()}")