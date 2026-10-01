# Preprocessing and Split Design

## Text preparation

The training script converts emoji to text descriptions when `demoji` is available, replaces URLs and mentions with placeholders, removes the `#` from hashtags, and collapses repeated whitespace. It preserves case, punctuation, digits, and symbols. It filters generator placeholder text, drops duplicate non-augmented sentences based on a normalized grouping key, and stores the cleaned model input in `model_text`.

The normalization used for grouping lowercases text and removes non-alphanumeric characters before comparison. For augmented rows, grouping uses `original_text` when available; otherwise it falls back to the row's text. This distinction is intended to keep variants of one base sentence together.

## Split method

The script uses `StratifiedGroupKFold` with seed `42`: one five-fold split to form training versus the remaining data, followed by a two-fold split of the remainder for validation and test. The saved run config records the seed, and the code asserts that group identifiers do not overlap between splits.

| Split | Rows | Label 0 | Label 1 | Unique groups |
|---|---:|---:|---:|---:|
| Train | 23,216 | 12,358 | 10,858 | 22,597 |
| Validation | 2,902 | 1,545 | 1,357 | 2,825 |
| Test | 2,902 | 1,545 | 1,357 | 2,824 |

The saved split files were checked: group overlap was zero for train/validation, train/test, and validation/test. See the actual [`train.csv`](../cyberbully_v0.1_run/train.csv), [`val.csv`](../cyberbully_v0.1_run/val.csv), and [`test.csv`](../cyberbully_v0.1_run/test.csv) artifacts.

## Interpretation

The held-out test set is a split of this project dataset, not an independently collected external test set. Grouping reduces one form of leakage between related original and augmented rows; it does not establish independence from source, topic, template, or generator characteristics.