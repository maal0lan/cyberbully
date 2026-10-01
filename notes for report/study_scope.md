# Study Scope

## System under study

The documented experiment fine-tunes `distilbert-base-uncased` for binary text classification. It also trains a 20-category auxiliary prediction head. The binary output is the primary task; the category output is a second supervised task.

## Project motivation: a pip-installable Python package

The broader engineering goal is to turn the detection work into a reusable Python package that users can install with `pip`, rather than requiring them to assemble scripts and model files themselves. The team's motivation is that existing packages did not meet the needs this project is trying to address. The repository does not include a systematic survey or benchmark of those packages, so this should be presented as project motivation, not as a general finding about the Python ecosystem.

The current repository has training and prediction scripts, but no standard package metadata (`pyproject.toml`, `setup.py`, or `setup.cfg`) was found at the project root. Package publication and pip installation are therefore goals, not capabilities demonstrated by the current artifacts. See [reproduction and artifacts](reproduction_and_artifacts.md).

## Candidate research questions

These are questions the available experiment can inform, not conclusions already established:

1. How does this model classify examples from the project's merged explicit and non-explicit text dataset under a grouped, stratified holdout split?
2. How do the fixed `0.5` decision threshold and the threshold selected on validation data differ on the held-out test split?
3. How does performance change on the included augmented examples and on programmatically perturbed test text?

The repository does not contain controlled ablations for the auxiliary category head, class weighting, or grouped versus random splitting. It therefore does not establish the causal contribution of any one design choice.

## Intended paper scope

Describe a project-specific dataset and one saved model run. Do not generalize the measured scores to all social-media users, platforms, languages, or real-world moderation settings. See [limitations](limitations_and_integrity.md) and [dataset provenance](data_provenance.md).