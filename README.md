# Cyberbullying Text Detection

## Project goal

This project detects cyberbullying in a single text input using a DistilBERT-based classifier.

It includes:
- `cyberbully_final.py`: training, evaluation, and model export logic
- `predict_cyberbullying_cli.py`: simple CLI for predicting one text at a time
- `cyberbully_model_final/`: saved model checkpoint and tokenizer files
- `requirements.txt`: package requirements for the Python environment
- `environment.yml`: optional Conda environment manifest
- `activate_cyberbully_env.bat`: optional Windows helper to activate the Conda environment

## Inputs

- For prediction: a single text string such as a comment, tweet, or message.
- For training/retraining: the dataset `cyberbullying_data.csv`.

## Outputs

- `predict_cyberbullying_cli.py` prints:
  - predicted label: `cyberbullying` or `not_cyberbullying`
  - probability score for the `cyberbullying` class
  - threshold used for classification
- `cyberbully_final.py` trains the model and saves:
  - `cyberbully_model_final/best_model.pt`
  - tokenizer files in `cyberbully_model_final/`
  - `cyberbully_model_final/config.json`

## Requirements

Install dependencies with pip:

```powershell
pip install -r requirements.txt
```

### Optional: Conda

Using Conda is optional. If you want to use it, create and activate the environment:

```powershell
conda env create -f environment.yml
activate_cyberbully_env.bat
```

If you do not want Conda, just use `pip install -r requirements.txt`.

## Using the prediction CLI

Run the CLI with a single text input:

```powershell
python predict_cyberbullying_cli.py "This is an example message"
```

If your model directory is named differently, provide it explicitly:

```powershell
python predict_cyberbullying_cli.py --model-dir cyberbully_model_final "This is an example message"
```

## Training or re-running the model

To train or evaluate the model and save the checkpoint, run:

```powershell
python cyberbully_final.py
```

This script:
- loads `cyberbullying_data.csv`
- trains a DistilBERT classifier
- saves the best checkpoint to `cyberbully_model_final/best_model.pt`
- saves tokenizer and config files

## Model evaluation and graphs

There is an evaluation helper in `1_final_model/evaluate_model.py` that can be used to run a full test inference and generate evaluation plots.

Running the evaluation script produces an `eval_results/` folder containing:
- `roc_curve.png` — Receiver Operating Characteristic (ROC) curve with AUC
- `pr_auc_curve.png` — Precision-Recall curve with AUC and baseline
- `accuracy_vs_threshold.png` — accuracy, F1, precision, and recall across decision thresholds
- `confusion_matrix.png` — confusion matrix showing true/false positives and negatives

These graphs help you:
- compare model performance across thresholds
- see how precision and recall trade off
- confirm whether the selected classification threshold is appropriate
- inspect error patterns with the confusion matrix

## Notes

- The CLI uses `best_model.pt` from `cyberbully_model_final/` by default.
- The prediction label threshold is loaded from `config.json` when available.
- If `demoji` is not installed, emojis are stripped instead of converted.
