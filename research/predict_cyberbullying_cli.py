"""
Single-text cyberbullying prediction CLI.

Usage:
    python predict_cyberbullying_cli.py "That message is awful"
    python predict_cyberbullying_cli.py --model-dir cyber_model_final "You are amazing"
"""

import argparse
import json
import os
import re

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

try:
    import demoji
    USE_DEMOJI = True
except ImportError:
    USE_DEMOJI = False

DEFAULT_MODEL_DIRS = ["cyber_model_final", "cyberbully_model_final"]
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 128
DEFAULT_THRESHOLD = 0.5


def clean_text(text: str) -> str:
    text = str(text)
    if USE_DEMOJI:
        text = demoji.replace_with_desc(text, sep=" ")
    else:
        text = text.encode("ascii", "ignore").decode("ascii")

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[url]", text)
    text = re.sub(r"@\w+", "[user]", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_model_dir(preferred_dir: str | None = None) -> str:
    if preferred_dir and os.path.isdir(preferred_dir):
        return preferred_dir

    for candidate in DEFAULT_MODEL_DIRS:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "No model directory found. Create one of: "
        + ", ".join(DEFAULT_MODEL_DIRS)
    )


def load_model(model_dir: str) -> tuple[torch.nn.Module, DistilBertTokenizerFast, float, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_dir if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
        else DEFAULT_BASE_MODEL
    )

    threshold = DEFAULT_THRESHOLD
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
            threshold = float(config.get("threshold", DEFAULT_THRESHOLD))

    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_BASE_MODEL, num_labels=2
    )

    state_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"Could not find best_model.pt in '{model_dir}'. "
            "Save your checkpoint there or point --model-dir to the correct folder."
        )

    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, threshold, device


@torch.no_grad()
def predict_text(text: str, model: torch.nn.Module, tokenizer: DistilBertTokenizerFast, threshold: float, device: torch.device) -> tuple[str, float]:
    inputs = tokenizer(
        clean_text(text),
        truncation=True,
        padding="max_length",
        max_length=DEFAULT_MAX_LEN,
        return_tensors="pt",
    )

    outputs = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    cyber_prob = float(probs[1].cpu())
    label = "cyberbullying" if cyber_prob >= threshold else "not_cyberbullying"
    return label, cyber_prob


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a single text is cyberbullying using a saved DistilBERT model."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model-dir",
        "-m",
        default=None,
        help="Directory containing best_model.pt. Defaults to cyber_model_final or cyberbully_model_final.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Override the saved threshold. Default is loaded from config.json or 0.5.",
    )

    args = parser.parse_args()
    text = args.text
    if not text:
        text = input("Enter text to classify: ").strip()

    model_dir = find_model_dir(args.model_dir)
    model, tokenizer, saved_threshold, device = load_model(model_dir)
    threshold = args.threshold if args.threshold is not None else saved_threshold

    label, probability = predict_text(text, model, tokenizer, threshold, device)

    print("\n=== Cyberbullying prediction ===")
    print(f"Model directory : {model_dir}")
    print(f"Text            : {text}")
    print(f"Prediction      : {label}")
    print(f"Probability     : {probability:.4f}")
    print(f"Threshold       : {threshold:.4f}")
    print("(Probability is for the cyberbullying class.)")


if __name__ == "__main__":
    main()
