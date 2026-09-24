"""
Download model files from Google Drive folders and preserve original Drive file names.

Usage:
    python download_drive_models.py

This script requires `gdown` and downloads the files from two Google Drive folders.
"""

import os
import subprocess
import sys

FOLDER_LINKS = [
    "https://drive.google.com/drive/folders/142uQgqE1JsclxDh-DL-ionCB8Flqbf8H?usp=sharing",
    "https://drive.google.com/drive/folders/1J0xiFBJxbwLREqR6WhPzaxd-PyNA28uQ?usp=sharing",
]

OUTPUT_DIR = "downloaded_models"


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is not installed. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download_folder(folder_url: str, output_dir: str) -> None:
    print(f"Downloading folder: {folder_url}")
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        folder_url,
        "-O",
        output_dir,
    ]
    subprocess.check_call(command)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_gdown()

    for idx, folder_url in enumerate(FOLDER_LINKS, start=1):
        folder_output = os.path.join(OUTPUT_DIR, f"folder_{idx}")
        os.makedirs(folder_output, exist_ok=True)
        download_folder(folder_url, folder_output)

    print(f"\nDownload complete. Files saved under: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
