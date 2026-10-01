# Limitations and Claim Boundaries

## Dataset and labels

- The dataset is project-specific and includes synthetic/generated and augmented examples. The repository does not establish that its distribution represents naturally occurring platform content.
- Category-derived binary labels encode the project's taxonomy. Evidence of complete human adjudication or inter-annotator agreement was not found.
- The project notes discuss judging and manual review, but open tasks remain in those notes. Do not present those tasks as completed quality controls without supporting records.
- English `distilbert-base-uncased` inputs and this dataset do not support claims about other languages.

## Evaluation

- The test data is held out from model training within the supplied dataset, but it is not an external-domain evaluation.
- The test split has 2,902 rows from one split seed. No confidence intervals, repeated-seed results, or independent replication are included in the cited run artifacts.
- The training metrics use a threshold selected on validation data. The separate evaluator's `0.35` threshold was selected on test labels and must not be used as a confirmatory headline result.
- Synthetic robustness results cover only the transformations implemented by this repository and include just 78 existing augmented test rows for one reported subset.
- Aggregate scores do not establish safety, fairness, calibrated risk, or suitability for automated moderation.

## Claims to avoid without new evidence

Do not claim state-of-the-art performance, real-world effectiveness, production readiness, comprehensive adversarial robustness, fairness across demographic groups, human-verified labels, causal benefit from an individual training technique, or superiority to the archived Keras models.

## Citation status

The inspected project documents do not provide a scholarly bibliography for the model, dataset, or evaluation methods. Add and verify primary academic references before submission; no external citations have been invented in this dossier.