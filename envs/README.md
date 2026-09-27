# Environments

Three separate environments; do not merge them.

| File | Used for | Python |
|---|---|---|
| `../requirements.txt` | Feature extraction, SVM classifiers, analysis | 3.10 |
| `requirements-mage.txt` | Longformer baseline (MAGE's training script) | 3.10 |
| `requirements-baselines.txt` | FastText and GLTR baselines | 3.11 |

The MAGE environment pins old versions because MAGE's `main.py` relies on APIs
removed in later releases. Install torch from the PyTorch index as noted at the
top of each file — the CUDA builds are not on PyPI, and `pip install -r` will
abort if they are listed there.