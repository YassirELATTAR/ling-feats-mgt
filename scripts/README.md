# Scripts

Shell wrappers for the pipeline. Run them from the repository root.

| Script | What it does |
|---|---|
| `run_elfen.sh` | Extract features from a single CSV. Usage: `./scripts/run_elfen.sh INPUT.csv OUTPUT_DIR` |
| `run_fasttext_grid.sh` | Exploratory hyperparameter grid for the FastText baseline. Not the reported configuration — see the paper appendix for that. |
| `run_gltr.sh` | GLTR feature extraction (GPT-2-XL, multi-GPU) followed by the grid-searched logistic regression. Requires `HF_HOME` to be set. |
