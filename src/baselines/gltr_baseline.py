"""
GLTR baseline (Gehrmann et al. 2019) reimplementation: GPT-2 token-rank histogram -> logistic regression.
Exact classifier used by MAGE on top of GLTR features is not released; this is the standard 4-bin variant. State it.



Step 1 (GPU): extract features for every csv you need
  CUDA_VISIBLE_DEVICES=0 python gltr_baseline.py extract --csv data/mage_raw_ext/cross_domains_cross_models/train.csv --out feats/cont/train.npz
Step 2 (CPU): train + evaluate
  python gltr_baseline.py eval --train feats/cont/train.npz --test feats/cont/test.npz feats/cont/test_ood_gpt.npz --out results/gltr_cont.json
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV  

BINS = [10, 100, 1000]          # rank buckets: <=10, <=100, <=1000, >1000


@torch.no_grad()
def extract(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained(a.model); tok.pad_token = tok.eos_token
    ## for the OOM error fix, we need to change to the one below
    lm = GPT2LMHeadModel.from_pretrained(a.model).to(dev).eval().half()

    # dtype = torch.float16 if a.fp16 else torch.float32
    # lm = GPT2LMHeadModel.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()

    df = pd.read_csv(a.csv)
    texts = [str(t) for t in df.text]
    ## new change  
    feats = np.zeros((len(texts), len(BINS)), dtype=np.float32)
    #feats = np.zeros((len(texts), 3), dtype=np.float32)  # MAGE setting: 3 bins only, no normalization
    order = np.argsort([len(t) for t in texts])            # length-sort for less padding

    for s in tqdm(range(0, len(texts), a.bs), desc=os.path.basename(a.csv)):
        idx = order[s:s + a.bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.max_len).to(dev)
        logits = lm(**enc).logits[:, :-1].float()           # predict token t+1 from 0..t
        target = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        tgt_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        rank = (logits > tgt_logit.unsqueeze(-1)).sum(-1) + 1   # 1 = most probable
        for j, i in enumerate(idx):
            r = rank[j][mask[j]].cpu().numpy()
            if len(r) == 0: continue
            feats[i] = np.array([(r <= 10).sum(), (r <= 100).sum(), (r <= 1000).sum()], dtype=np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=feats, y=df.label.values.astype(int))
    print("saved", a.out, feats.shape)


def metrics(y, p_human, name):
    pred = (p_human >= 0.5).astype(int)
    hr = (pred[y == 1] == 1).mean(); mr = (pred[y == 0] == 0).mean()
    r = {"testset": name, "n": int(len(y)), "HumanRec": hr, "MachineRec": mr, "AvgRec": (hr + mr) / 2,
         "AUROC": roc_auc_score(y, p_human), "F1_macro": f1_score(y, pred, average="macro")}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}); return r

def evaluate(a):
    tr = np.load(a.train); sc = StandardScaler().fit(tr["X"]); Xtr = sc.transform(tr["X"])
    grid = {"solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.5]}                          # only used by elasticnet
    gs = GridSearchCV(LogisticRegression(max_iter=2000), grid, cv=5, scoring="roc_auc",
                      n_jobs=-1, error_score=np.nan).fit(Xtr, tr["y"])   # invalid solver/penalty combos -> nan, skipped
    print("best:", gs.best_params_, round(gs.best_score_, 4))
    clf = gs.best_estimator_
    res = []
    for t in a.test:
        d = np.load(t); p_human = clf.predict_proba(sc.transform(d["X"]))[:, 1]
        res.append(metrics(d["y"], p_human, os.path.basename(t)))
    res.append({"best_params": gs.best_params_, "cv_auroc": gs.best_score_})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=str)


if __name__ == "__main__":
    # some arguments in default as still in default and changed in the bash script to match the MAGE setting
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--csv", required=True); e.add_argument("--out", required=True)
    e.add_argument("--model", default="gpt2-xl"); e.add_argument("--bs", type=int, default=8); e.add_argument("--max_len", type=int, default=1024); e.add_argument("--fp16", action="store_true")
    v = sub.add_parser("eval"); v.add_argument("--train", required=True); v.add_argument("--test", nargs="+", required=True); v.add_argument("--out", required=True)
    a = p.parse_args(); extract(a) if a.cmd == "extract" else evaluate(a)