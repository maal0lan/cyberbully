# Reproduction and Artifacts

## Training entry point

The training script is [`cyberbully_final.py`](../cyberbully_final.py). Its saved run configuration records the dataset path, model identifier, seed, maximum length, batch size, epochs, learning rate, warmup, weight decay, patience, auxiliary-loss weight, and label source. The trained checkpoint and tokenizer are retained in `cyberbully_v0.1_run`.

From the repository root, the documented default training command is:

```powershell
python cyberbully_final.py
```

This command can retrain and overwrite artifacts in the default run directory. Do not rerun it merely to inspect this dossier.

## Evaluation entry point

The standalone evaluator is [`eval_results/evaluate_models.py`](../eval_results/evaluate_models.py). It reloads the saved PyTorch checkpoint and defaults to the saved test CSV. Its test-threshold optimization caveat is documented in [evaluation](evaluation_results.md).

## Package status

The intended deliverable is a reusable Python package installable with `pip`. At the time these notes were prepared, the repository had not yet added standard packaging metadata or a package layout, so no `pip install` workflow is documented as working. The current scripts and `requirements.txt` are not, by themselves, evidence that this project is an installable distribution.

## Evidence copies

Copies of source JSON reports and figures are stored under [`assets`](source_register.md). They were copied without changing their contents; the source files remain in their original locations. The model weight files were not duplicated because the request concerned report evidence, and the checkpoint is already present in the repository.