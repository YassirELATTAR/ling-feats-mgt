# Baselines

We re-train three of the four detectors reported by MAGE (Li et al., 2024):
FastText, GLTR and Longformer. DetectGPT is omitted because it needs
token-level likelihoods from each generating model, which are unavailable for
the commercial models in our extended data.

## Longformer

The Longformer detector uses MAGE's own training script unmodified. Clone their
repository and point `MAGE_REPO` at it:

    git clone https://github.com/yafuly/MAGE.git
    export MAGE_REPO=/path/to/MAGE

`training/longformer/main.py` in that repository is the script we call; the
hyperparameters are theirs (`training/longformer/train.sh`): max sequence length
2048, effective batch size 16, learning rate 3e-5, 5 epochs, fp16, seed
42629309. 


Test-bed CSVs are built with MAGE's `deployment/prepare_testbeds.py`; the edits
we applied to it are documented in `prepare_testbeds_patch.md`.

## FastText and GLTR

MAGE does not release code for these, so `fasttext_baseline.py` and
`gltr_baseline.py` are our re-implementations of the setup described in their
paper (Appendix). Where a hyperparameter is unspecified we use the library
default; see the paper's appendix for the resulting deviations.

## Evaluation

`eval_mage.py` computes HumanRec, MachineRec, AvgRec, AUROC and macro F1 from
the Longformer prediction files, under both the default decision rule and
MAGE's refined out-of-distribution threshold.

MAGE repository: https://github.com/yafuly/MAGE