#!/bin/bash
DATA="data/mage_raw_ext/cross_domains_cross_models" #Example
OUT="results/ft_grid"
EPOCHS="5 10 25"
LRS="0.1 0.5 1.0"
NGRAMS="1 2 3"
THREADS=16

SCRIPT=src/baselines/fasttext_baseline.py

mkdir -p $OUT
for ep in $EPOCHS; do
  for lr in $LRS; do
    for ng in $NGRAMS; do
      name="e${ep}_lr${lr}_ng${ng}"
      echo "=== $name ==="
      python $SCRIPT --data $DATA --out $OUT/$name \
        --epoch $ep --lr $lr --ngrams $ng --threads $THREADS \
        > $OUT/$name.log 2>&1
      grep -E "^valid|'test'" $OUT/$name.log
    done
  done
done

echo; echo "=== SUMMARY (AvgRec / AUROC on test) ==="
for f in $OUT/*.log; do
  printf "%-22s " $(basename $f .log); grep "'test'" $f | grep -oE "'AvgRec': [0-9.]+, 'AUROC': [0-9.]+"
done