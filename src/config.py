"""Central configuration loader.

Everything path-related lives in configs/paths.yaml. Import `CFG` for the raw
dict, or call load_config(path) to load a different file.

    from src.config import CFG, resolve
    features = resolve(CFG['mage_features'])
"""
import json
import os
from pathlib import Path

import yaml

# Repository root = parent of the directory holding this file.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


def load_config(path=None):
    """Load paths.yaml. Override with the LFMGT_CONFIG env var or an argument."""
    path = Path(path or os.environ.get("LFMGT_CONFIG", DEFAULT_CONFIG))
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()


def resolve(p):
    """Make a config path absolute, relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_json(key):
    return json.load(open(resolve(CFG[key])))


def model_families():
    """{family: [model_tag, ...]}"""
    return _load_json("model_families_file")


def selected_features():
    """{area: [feature, ...]} — the reduced (~90) feature set."""
    return _load_json("selected_features_file")


def feature_groups():
    """{area: [feature, ...]} — feature areas used for the ablation study."""
    raw = _load_json("feature_groups_file")
    # the consistency report stores {area: {"features": [...], ...}}
    return {k: (v["features"] if isinstance(v, dict) else v) for k, v in raw.items()}