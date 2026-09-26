"""
python eval_mage.py output_cont_42629309_lfbase_ddp \
    ../data/mage_raw_ext/cross_domains_cross_models/test.csv




# The main needs train/ valid files for the schema if I want to use a prediction head (e.g., for the MAGE setting). The following is an example of how to run the main script to get predictions on the OOD test set, and then evaluate them with this script.

D=./data/mage_raw_ext/cross_domains_cross_models
CUDA_VISIBLE_DEVICES=8 python3 main.py \
  --do_predict \
  --model_name_or_path ./output_cont_42629309_lfbase_ddp \
  --train_file $D/train.csv --validation_file $D/valid.csv \
  --test_file ./data/mage_raw_ext/test_ood_all.csv \
  --max_seq_length 2048 --per_device_eval_batch_size 16 --fp16 \
  --output_dir ./output_cont_42629309_lfbase_ddp/pred_ood






python eval_mage.py output_cont_42629309_lfbase_ddp/pred_ood ./data/mage_raw_ext/test_ood_all.csv
python eval_mage.py output_cont_42629309_lfbase_ddp/pred_ood ./data/mage_raw_ext/test_ood_all.csv --th -3.08583984375


"""

import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

import argparse
p = argparse.ArgumentParser()
p.add_argument("out_dir"); p.add_argument("test_csv")
p.add_argument("--th", type=float, default=None)
a = p.parse_args()
out_dir, test_csv, th = a.out_dir, a.test_csv, a.th

# out_dir, test_csv = sys.argv[1], sys.argv[2]
# th = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--th" else None

logits = np.loadtxt(f"{out_dir}/predict_results_probs.csv", delimiter=",")
y = pd.read_csv(test_csv)["label"].values.astype(int)          # 1 = human, 0 = machine
assert len(y) == len(logits), f"rows mismatch: {len(y)} labels vs {len(logits)} predictions"

score = logits[:, 0] - logits[:, 1]                             # >0 -> machine (as in MAGE utils.detect)
if th is None:
    pred = logits.argmax(1)                                     # 1 = human
else:
    pred = (score <= th).astype(int)                            # machine if score > th, else human

human_rec   = (pred[y == 1] == 1).mean()
machine_rec = (pred[y == 0] == 0).mean()
res = {"test_csv": test_csv, "threshold": th, "n": int(len(y)),
       "HumanRec": human_rec, "MachineRec": machine_rec, "AvgRec": (human_rec + machine_rec) / 2,
       "AUROC": roc_auc_score(y, -score), "F1_macro": f1_score(y, pred, average="macro")}
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

tag = "argmax" if th is None else f"th{th}"
json.dump(res, open(f"{out_dir}/mage_metrics_{tag}.json", "w"), indent=2)