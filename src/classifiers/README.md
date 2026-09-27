# Classifiers

Linear SVM classifiers over the extracted linguistic features. All paths,
domains, model families and SVM settings come from `configs/paths.yaml` and the
JSON files next to it.

| Script | What it does |
|---|---|
| `classifier_core.py` | The classifier itself (`TestbedClassifier`), the testbed definitions (`MAGETestbedConfig`), and the runner for all eight testbeds. Also the main entry point. |
| `ablation_study.py` | Removes one feature area at a time and re-runs a chosen testbed. |
| `cumulative_ablation.py` | Removes feature areas progressively, in the order given by the single-area ablation. |
| `cross_cmv_mage_experiment.py` | Cross-dataset experiments between MAGE's CMV domain (continuation prompts) and the CMV corpus of Dönmez & Falenska (2025) (direct-response prompts). |

## Running

Edit the configuration block at the top of each `__main__` and run from the
repository root:

```bash
python src/classifiers/classifier_core.py          # main testbeds -- Could also be used for the per-feature-area analysis (e.g., lexical_richness only)
python src/classifiers/ablation_study.py           # single-area ablation
python src/classifiers/cumulative_ablation.py      # cumulative ablation
python src/classifiers/cross_cmv_mage_experiment.py
```

In `classifier_core.py` the relevant settings are:

- `RUN_NAME` — the subfolder under `results/` and `logs/` for this run
- `RUN_TESTBEDS` — which testbeds to run (`'1'`, `'1_1'`, `'2'` … `'8'`)
- `feature_groups` — `['combined']` for the full 284-feature set, or a single
  area such as `['lexical_richness']`
- `filter_90_features` — use the reduced feature set from
  `configs/selected_features.json` instead

In `ablation_study.py`, `TESTBEDS` selects the testbeds, `RUN_ONLY` restricts
the run to specific feature areas, and `SPECIFIC_PAIRS` limits testbed 8 to a
subset of domain–model pairs.

## Output

Each classifier writes to
`results/<RUN_NAME>/<testbed-group>/<testbed-name>/<feature-group>/`:

- `results_<timestamp>.txt` — all metrics, per-class scores, confusion matrix
- `top_50_features_<timestamp>.csv` and `all_features_<timestamp>.csv` — SVM
  coefficients
- one `global_results_<timestamp>.json` per run, summarising every classifier

Console output is also teed to `logs/<RUN_NAME>/`.

## Scale

Testbed 1 trains 270 classifiers (one per domain–model pair), testbeds 1.1 and
8 train 70 each, the rest between 1 and 10. A full ablation multiplies that by
the number of feature areas, so start with a single testbed to gauge the
runtime.