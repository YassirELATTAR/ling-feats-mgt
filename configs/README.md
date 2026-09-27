# Configuration

| File | What it is |
|---|---|
| `paths.yaml` | Root directories (data, features, results, logs), the domain and model lists, and the SVM hyperparameters. Edit this first if your data lives elsewhere. |
| `model_families.json` | Mapping of model family → individual model tags. The tags must match the folder names under `data/{split}/AI/{domain}/{family}/`. |
| `selected_features.json` | The reduced (~90) feature set used when a script is run with `filtered_90=True`. |
| `feature_groups.json` | The feature areas used by the ablation study: area → list of feature names. |

`feature_groups.json` is **generated**, not hand-written: it is a copy of the
report produced by `src/features/check_features_consistency.py`. If you
re-extract features, regenerate it rather than editing it by hand, or the
ablation groups will no longer match the columns in your feature files.