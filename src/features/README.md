# Feature extraction

Linguistic features are extracted with
[ELFEN](https://github.com/mmmaurer/elfen) (v1.1.9) using spaCy's
`en_core_web_lg`. Eleven feature areas are extracted separately and also as one
combined file (284 features in total).

| Script | What it does |
|---|---|
| `elfen_extractor.py` | Extract all feature areas from one CSV. Token normalisation is the default and is what the paper uses. |
| `run_multiple_extractors.py` | Batch version: walks a directory tree of CSVs and mirrors it under the output root. |
| `check_features_consistency.py` | Verify that every feature file has the same columns, and write `feature_consistency_report.json` (the source of `configs/feature_groups.json`). |
| `check_nan_values.py` | Report NaN and Inf values per feature area, with a suggested imputation method for each affected column. Reporting only; nothing is modified. |

Typical order:

```bash
python src/features/run_multiple_extractors.py --input data/train/AI/wp --output data_features_prepared --normalize token
python src/features/check_features_consistency.py --root data_features_prepared
python src/features/check_nan_values.py --root data_features_prepared
```

The feature selection per area, and the rationale for it, is described in the
paper's appendix.