# Labels and Taxonomy

## Binary target

The training code maps the 20 semantic categories to two labels. Label `0` means not cyberbullying; label `1` means cyberbullying. With the saved run configuration's `label_source: category`, the training pipeline derives labels from this mapping. If a category is unknown to the mapping, the code retains its raw `gen_label` value and logs the unknown category.

| Label 0: not cyberbullying | Label 1: cyberbullying |
|---|---|
| `venting_no_target` | `threat`, `threat_clean` |
| `topic_opinion_negative` | `targeted_insult_identity`, `targeted_insult_general` |
| `sarcasm_among_friends` | `explicit_insult_clean`, `backhanded_compliment` |
| `meta_commentary_condemning_hate` | `manipulation_gaslighting`, `social_exclusion` |
| `meta_commentary_condemning_bullying` | `pile_on_mob`, `implicit_mockery` |
| `supportive_message`, `neutral_statement` | `rumor_spreading` |
| `genuine_compliment`, `constructive_criticism` | |

The mapping is implemented in [`cyberbully_final.py`](../cyberbully_final.py). The dataset's `category` values encode a project-defined taxonomy; the repository does not include an independent annotation handbook or inter-annotator agreement study.

## Relabeling and review

The pipeline can write rows whose raw label differs from the category-derived label to `relabel_review.csv`. The existence of this review file records candidates for inspection, not that all rows were manually reviewed or adjudicated. Do not describe category-derived labels as human-validated unless additional evidence is added.

## Scope caveat

The labels represent the project taxonomy, not a universal definition of cyberbullying. For discussion of ambiguous categories and context limits, see [limitations](limitations_and_integrity.md).