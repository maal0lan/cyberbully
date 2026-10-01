# Cyberbullying Research Dossier

This folder organizes the repository evidence for a research paper and the broader project goal: build a reusable Python package for cyberbullying detection that users can install with `pip`. The project motivation is that existing packages did not fit the team's intended needs; this is a project rationale, not a systematic comparison of available packages. The current repository is still script-based and does not yet contain standard package-install metadata. These notes describe the experiment represented by `cyberbully_v0.1_run`; they do not retrain or replace that model.

## Topic map

- [Study scope and research questions](study_scope.md)
- [Dataset profile](dataset_profile.md)
- [Labels and taxonomy](label_schema.md)
- [Dataset provenance](data_provenance.md)
- [Preprocessing and split design](preprocessing_and_split.md)
- [Model and training setup](model_training.md)
- [Held-out evaluation](evaluation_results.md)
- [Robustness results](robustness_and_errors.md)
- [Legacy model reports](legacy_model_reports.md)
- [Limitations and claim boundaries](limitations_and_integrity.md)
- [Reproduction and artifacts](reproduction_and_artifacts.md)
- [Source register and copied evidence](source_register.md)

## At-a-glance result

The saved DistilBERT run uses a threshold selected on validation data (`0.23`). On the 2,902-row held-out test split, its saved report records accuracy `0.9714`, macro-F1 `0.9713`, ROC-AUC `0.9964`, and PR-AUC `0.9958` at that threshold. At the fixed `0.5` threshold, the report records accuracy `0.9728` and macro-F1 `0.9727`. The two operating points are reported separately; see [evaluation](evaluation_results.md) for counts and caveats.

## Evidence policy

Numbers and implementation details here are taken from saved artifacts or executable code. Planning documents are treated as plans, not proof that every planned step was completed. The copied JSON and image files are in [assets](source_register.md). External scholarly citations, human annotation evidence, and evaluation on independently collected data were not found in the inspected repository materials and are not asserted here.