#!/bin/bash
# run_gltr.sh — GLTR baseline (GPT-2-XL rank counts -> grid-searched LR), MAGE setting.
# Edit the SETTINGS block, then: bash run_gltr.sh
set -e

# ---------------- SETTINGS ----------------
BASE=data
FEATS=feats
RES=results
GPUS=(5 6 7)                  # free GPUs; needs >= 1 per concurrent job
BS=8
MAXLEN=1024
MODEL=$HF_HOME/gpt2-xl
SCRIPT=src/baselines/gltr_baseline.py

declare -A DATA=(
  [orig]="$BASE/mage_raw"
  [cont]="$BASE/mage_raw_ext"
)
FILES="cross_domains_cross_models/train.csv cross_domains_cross_models/test.csv"
OOD_CSV="$BASE/mage_raw_ext/test_ood_all_src.csv"     # same OOD set for both settings
# ------------------------------------------

mkdir -p $FEATS $RES $FEATS/shared
[ -d "$MODEL" ] || { echo "model dir not found: $MODEL"; exit 1; }

# ---- pre-flight: check everything BEFORE launching any job ----
for setting in "${!DATA[@]}"; do
  for f in $FILES; do
    [ -f "${DATA[$setting]}/$f" ] || { echo "missing ${DATA[$setting]}/$f"; exit 1; }
  done
done
[ -f "$OOD_CSV" ] || { echo "missing $OOD_CSV"; exit 1; }
echo "all input files present"

# ---- 1) feature extraction, one GPU per job, in parallel ----
g=0
launch () {   # $1=csv  $2=out  $3=logname
  [ -f "$2" ] && { echo "skip $2"; return; }
  gpu=${GPUS[$((g % ${#GPUS[@]}))]}; g=$((g+1))
  echo "GPU $gpu -> $3"
  CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT extract --csv "$1" --out "$2" \
    --bs $BS --max_len $MAXLEN --model $MODEL > $FEATS/$3.log 2>&1 &
}

for setting in "${!DATA[@]}"; do
  mkdir -p $FEATS/$setting
  for f in $FILES; do
    launch "${DATA[$setting]}/$f" "$FEATS/$setting/$(basename $f .csv).npz" "$setting.$(basename $f .csv)"
  done
done
launch "$OOD_CSV" "$FEATS/shared/test_ood_all.npz" "shared.test_ood_all"

wait
echo "=== extraction done ==="

# ---- 2) train LR (grid search, 5-fold CV) + evaluate (CPU only) ----
for setting in "${!DATA[@]}"; do
  echo "=== $setting ==="
  python $SCRIPT eval --train $FEATS/$setting/train.npz \
    --test $FEATS/$setting/test.npz $FEATS/shared/test_ood_all.npz \
    --out $RES/gltr_$setting.json
done
echo "=== done: $RES/gltr_*.json ==="

