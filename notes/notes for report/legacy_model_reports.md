# Legacy Model Reports

The repository contains separate Keras artifacts and JSON metrics under `cyberbully_output`. These are not the DistilBERT run and should not be blended into its results.

| Copied report | Stored description | Reported accuracy | Reported F1 | Provenance caveat |
|---|---|---:|---:|---|
| [metrics.json](assets/json/legacy_keras/metrics.json) | Model type is not stated in this JSON | 0.8691 | 0.9262 | The JSON does not identify the exact checkpoint, split, or metric protocol. |
| [metrics_pure_dataset.json](assets/json/legacy_keras/metrics_pure_dataset.json) | `Keras_BiLSTM` | 0.8544 | 0.9167 | This JSON identifies a model type but does not include enough split provenance for a controlled comparison. |

These reports are preserved for auditability, not presented as a valid benchmark against the grouped-split PyTorch model. Their thresholds differ, and the repository materials inspected here do not establish matched data, split, or evaluation procedures. No claim that one architecture is superior should be based on these values alone.