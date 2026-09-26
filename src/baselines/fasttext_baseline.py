"""
FastText baseline on a MAGE testbed folder (train.csv / valid.csv / test.csv, columns text,label; 0=machine 1=human).
Hyperparameters not given in MAGE paper -> library defaults + wordNgrams=2 (state this in the paper).

Usage:
  python fasttext_baseline.py --data data/mage_raw_ext/cross_domains_cross_models --out results/fasttext_maxmin52 --maxn 5 --minn 2

  python fasttext_baseline.py --data data/mage_raw/cross_domains_cross_models --out results/fasttext_mage_org


  MAGE DEFAULT:
  python fasttext_baseline.py --data data/mage_raw_ext/cross_domains_cross_models --out results/fasttext_new_ext_e100 --epoch 100 --ngrams 2 --extra_test data/mage_raw_ext/test_ood_all.csv


  # extra test sets (TB7 etc.):
  python fasttext_baseline.py --data ... --out ... --extra_test data/mage_raw_ext/test_ood_all.csv
"""
import argparse, os, re, json
import numpy as np, pandas as pd, fasttext
from sklearn.metrics import roc_auc_score, f1_score 


def to_ft(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(df.text, df.label):
            t = re.sub(r"\s+", " ", str(t)).strip()
            f.write(f"__label__{int(l)} {t}\n")


def evaluate(model, df, name):
    texts = [re.sub(r"\s+", " ", str(t)).strip() for t in df.text]
    labels, probs = model.predict(texts, k=2)
    # score = P(human)
    p_human = np.array([dict(zip(l, p))["__label__1"] for l, p in zip(labels, probs)])
    y = df.label.values.astype(int)
    pred = (p_human >= 0.5).astype(int)
    hr = (pred[y == 1] == 1).mean(); mr = (pred[y == 0] == 0).mean()
    res = {"testset": name, "n": len(y), "HumanRec": hr, "MachineRec": mr,
        "AvgRec": (hr + mr) / 2, "AUROC": roc_auc_score(y, p_human),
        "F1_macro": f1_score(y, pred, average="macro")}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return res


def main(a):
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv(f"{a.data}/train.csv"); va = pd.read_csv(f"{a.data}/valid.csv"); te = pd.read_csv(f"{a.data}/test.csv")
    to_ft(tr, f"{a.out}/train.txt"); to_ft(va, f"{a.out}/valid.txt")
    print(f"train {len(tr)}  valid {len(va)}  test {len(te)}")
  
    model = fasttext.train_supervised(input=f"{a.out}/train.txt", epoch=a.epoch, lr=a.lr,
                                  wordNgrams=a.ngrams, dim=a.dim, minn=a.minn, maxn=a.maxn,
                                  thread=a.threads, verbose=2)
    model.save_model(f"{a.out}/model.bin")
    print("valid:", model.test(f"{a.out}/valid.txt"))

    results = [evaluate(model, te, "test")]
    for p in a.extra_test:
        results.append(evaluate(model, pd.read_csv(p), os.path.basename(p)))
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra_test", nargs="*", default=[])
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--ngrams", type=int, default=2)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--minn", type=int, default=0)
    p.add_argument("--maxn", type=int, default=0)
    main(p.parse_args())