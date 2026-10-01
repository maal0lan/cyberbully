# Source Register and Evidence Copies

## Copied JSON

| Dossier copy | Original source |
|---|---|
| [Dataset summary](assets/json/dataset/summary_statistics.json) | `dataset_generation/helper_files/dataset/analytics/summary_statistics.json` |
| [Training metrics](assets/json/training_run/metrics.json) | `cyberbully_v0.1_run/metrics.json` |
| [Training configuration](assets/json/training_run/run_config.json) | `cyberbully_v0.1_run/run_config.json` |
| [Standalone evaluation metrics](assets/json/independent_evaluation/pytorch_best_model_metrics.json) | `eval_results/pytorch_best_model_metrics.json` |
| [Legacy metrics](assets/json/legacy_keras/metrics.json) | `cyberbully_output/metrics.json` |
| [Pure-dataset Keras metrics](assets/json/legacy_keras/metrics_pure_dataset.json) | `cyberbully_output/metrics_pure_dataset.json` |

## Copied figures

Dataset figures: [label distribution](assets/images/dataset/01_label_distribution.png), [category distribution](assets/images/dataset/02_category_distribution.png), [unigrams](assets/images/dataset/03_word_frequency_unigrams.png), [bigrams](assets/images/dataset/04_word_frequency_bigrams.png), [text lengths](assets/images/dataset/05_text_length_and_tokens.png), [augmentation](assets/images/dataset/06_augmentation_analysis.png), and [target types](assets/images/dataset/07_target_type_distribution.png).

Model figures: [training history](assets/images/model/training_history.png), [training-run confusion matrix](assets/images/model/training_run_confusion_matrix.png), [ROC curve](assets/images/model/pytorch_best_model_roc_curve.png), [precision-recall curve](assets/images/model/pytorch_best_model_pr_auc_curve.png), [threshold sweep](assets/images/model/pytorch_best_model_accuracy_vs_threshold.png), [default-threshold confusion matrix](assets/images/model/pytorch_best_model_confusion_matrix_default.png), [test-selected-threshold confusion matrix](assets/images/model/pytorch_best_model_confusion_matrix_best_macro_f1.png), and [calibration curve](assets/images/model/pytorch_best_model_calibration_curve.png).

## Primary implementation and project sources

- [`cyberbully_final.py`](../cyberbully_final.py): cleaning, category mapping, grouping, split, architecture, training, threshold selection, and test reporting.
- [`evaluate_models.py`](../eval_results/evaluate_models.py): separate test evaluation and plot generation.
- [`model_training_specs.md`](../dataset_generation/helper_files/dataset/analytics/model_training_specs.md) and [summary statistics](../dataset_generation/helper_files/dataset/analytics/summary_statistics.json): dataset description and analytics; the prose specification includes recommendations that are not necessarily measured results.
- [`merge_datasets.py`](../dataset_generation/helper_files/expander/merge_datasets.py): explicit/non-explicit dataset merge implementation.
- [README](../README.md), [implementation notes](../cyberbullying_implementation.md), and the older [pipeline overview](../notes/PIPELINE_EXPLAINED.md): retained for context, but contain stale paths, planned work, or claims not independently supported by the current saved run. Prefer the code and copied run artifacts when they differ.

Copied files are byte-for-byte copies of the source artifacts. JSON is retained as evidence, including legacy outputs whose model/split provenance is incomplete; see [legacy reports](legacy_model_reports.md).