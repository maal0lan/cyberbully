import os
LANGUAGE_MAP = {
    'ar': 'Arabic',
    'zh': 'Chinese',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'eo': 'Esperanto',
    'fil': 'Filipino',
    'fi': 'Finnish',
    'fr': 'French',
    'fr-CA-u-sd-caqc': 'French (CA)',
    'de': 'German',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'it': 'Italian',
    'ja': 'Japanese',
    'kab': 'Kabyle',
    'tlh': 'Klingon',
    'ko': 'Korean',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'th': 'Thai',
    'tr': 'Turkish'
}

def rename_language_files(target_directory):
    # Ensure the directory exists
    if not os.path.exists(target_directory):
        print(f"Directory '{target_directory}' does not exist.")
        return

    # Sort keys by length descending to prevent partial matching conflicts (e.g., 'fr' matching 'fr-CA...')
    sorted_codes = sorted(LANGUAGE_MAP.keys(), key=len, reverse=True)

    for filename in os.listdir(target_directory):
        file_path = os.path.join(target_directory, filename)
        
        # Skip directories, process only files
        if not os.path.isfile(file_path):
            continue
            
        base_name, extension = os.path.splitext(filename)
        
        # Check if any language code is present in the filename
        for code in sorted_codes:
            # Matches code if it's the exact name, or separated by spaces/underscores/hyphens
            if code == base_name or f"_{code}" in base_name or f"-{code}" in base_name or f" {code}" in base_name:
                language_name = LANGUAGE_MAP[code]
                
                # Clean up characters like parentheses for filesystem safety if needed
                # (French (CA) is usually fine, but you can change it if your OS objects)
                safe_language_name = language_name.replace(" ", "_").replace("(", "").replace(")", "")
                
                new_filename = f"{safe_language_name}{extension}"
                new_file_path = os.path.join(target_directory, new_filename)
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
                
                break # Move to the next file once matched

# Replace '.' with the actual path to your folder if the files aren't in the same directory as the script
folder_path = '.' 
rename_language_files(folder_path)