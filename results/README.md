# Results

Aggregated results for every experiment reported in the paper. Each row is one
trained classifier. Full per-classifier output (one folder per classifier, with
the metrics file and the SVM coefficients) is not committed — rerun the scripts
in `src/classifiers/` to regenerate it.

| File | What it holds |
|---|---|
| `all_metrics_main_experiment.csv` | The eight testbeds (TB1–TB8) with the full 284-feature set. |
| `all_metrics_ablation.csv` | Same testbeds with one feature area removed at a time. |
| `all_metrics_lexical_richness.csv` | TB1.1 and TB8 using only the three lexical-richness features. |
| `cumulative_ablation_TB4.json` | Feature areas removed progressively on TB4, in order of single-area impact. |
| `cumulative_ablation_TB7.json` | The same procedure on TB7. |

## CSV columns

| Column | Meaning |
|---|---|
| `testbed` | Classifier name, e.g. `8_unseen_domain_model_pair_xsum_llama` |
| `feature_group` | Feature file used (`combined` = all 284 features) |
| `ablated_group` | Feature area removed (ablation file only) |
| `accuracy`, `auroc`, `avg_precision` | Overall metrics |
| `f1_macro`, `f1_micro`, `f1_weighted` | F1 variants; macro F1 is the primary metric |
| `avg_recall` | Mean of the two class recalls |
| `precision_human`, `recall_human`, `f1_human` | Human class (label 0) |
| `precision_ai`, `recall_ai`, `f1_ai` | AI class (label 1) |
| `tn`, `fp`, `fn`, `tp` | Confusion matrix |
| `path` | Source metrics file |

Macro F1 is the primary metric because test sets are not balanced in every
testbed; accuracy is reported for completeness.

## JSON structure

```json
{
  "ablation_order": ["morphological", "psycholinguistic", "..."],
  "all_results": [
    {"removed_groups": ["morphological"], "num_groups_removed": 1,
     "testbed": "...", "accuracy": 0.0, "auroc": 0.0, "f1_macro": 0.0}
  ]
}
```

`ablation_order` is the removal order, derived from the single-area ablation
(least harmful first). Each entry is one step: the areas removed so far and the
resulting performance. Only the three summary metrics are stored here; per-class
metrics for these runs are in the per-classifier output.

## Reproducing

```bash
python src/classifiers/classifier_core.py          # main experiment
python src/classifiers/ablation_study.py           # single-area ablation
python src/classifiers/cumulative_ablation.py      # cumulative ablation
```

Set `feature_groups = ['lexical_richness']` in `classifier_core.py` for the
lexical-richness run. The notebooks in `notebooks/` build the paper's tables and
figures from these files.