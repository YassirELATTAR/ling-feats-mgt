"""
Unseen-domain test set generation via API (GPT-5.6-sol).
Prompts follow MAGE Figure 11, one template per domain.

  export OPENAI_API_KEY=...
  python generate_unseen.py -m gpt-5.6-sol --data-root data

Reads  {data_root}/test/unseen/Human/{domain}/{domain}_human.csv
Writes {data_root}/test/unseen/AI/{domain}/{domain}_{model_tag}.csv  (doc_id,text,label)
Resume-safe: finished files are skipped, interrupted ones continue from a checkpoint.
"""
import argparse
import os
import random
import re
import time

import polars as pl
from openai import OpenAI

DOMAINS = ["cnn", "dialogsum", "imdb", "pubmed"]


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def build_prompt(domain, text):
    """Return (prompt, seed_to_prepend or None), following MAGE Figure 11."""
    sents = split_sentences(text)
    if domain == "cnn":
        return f"Write a news article given the following highlights: {' '.join(sents[:3])}", None
    if domain == "imdb":
        return f"Write a short movie review with the following beginning: {sents[0]}", None
    if domain == "dialogsum":
        turns = re.findall(r"(#Person\d+#:.*?)(?=#Person\d+#:|$)", text, flags=re.S)
        seed = " ".join(t.strip() for t in turns[:2]) if len(turns) >= 2 else " ".join(sents[:2])
        return f"Continue the following daily dialogue: {seed}", seed
    if domain == "pubmed":
        # MAGE prompts with the original question, which the human file does not carry;
        # we continue the beginning of the abstract instead (documented deviation).
        seed = " ".join(sents[:2])
        return f"Continue the following biomedical answer to a question: {seed}", seed
    raise ValueError(domain)


def generate(client, model, prompt, max_tokens, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def generate_domain(args, client, model_tag, domain):
    in_path = os.path.join(args.data_root, "test", "unseen", "Human", domain, f"{domain}_human.csv")
    if not os.path.exists(in_path):
        print(f"MISSING: {in_path}")
        return

    out_dir = os.path.join(args.data_root, "test", "unseen", "AI", domain)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{domain}_{model_tag}.csv")
    ckpt_path = os.path.join(out_dir, f"{domain}_{model_tag}_checkpoint.csv")

    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        return

    texts = [str(t) for t in pl.read_csv(in_path)["text"].to_list()]
    rng = random.Random(args.seed)
    if len(texts) > args.n_samples:
        texts = rng.sample(texts, args.n_samples)

    rows, start = [], 0
    if os.path.exists(ckpt_path):
        rows = pl.read_csv(ckpt_path).to_dicts()
        start = len(rows)
        print(f"resuming {domain} at {start}/{len(texts)}")

    print(f"\n=== {domain}: {len(texts)} samples | {args.model} ===", flush=True)
    for i in range(start, len(texts)):
        prompt, seed = build_prompt(domain, texts[i])
        out = generate(client, args.model, prompt, args.max_tokens)
        if out is None:
            pl.DataFrame(rows).write_csv(ckpt_path)
            raise SystemExit(f"failed at sample {i + 1}; checkpoint written, rerun to resume")
        full = f"{seed} {out}" if seed and not out.startswith(seed[:30]) else out
        rows.append({"doc_id": f"{model_tag}_{i + 1}", "text": full, "label": 1})
        if (i + 1) % args.save_every == 0:
            pl.DataFrame(rows).write_csv(ckpt_path)
            print(f"{i + 1}/{len(texts)}", flush=True)

    pl.DataFrame(rows).write_csv(out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"saved {len(rows)} -> {out_path}", flush=True)


def main(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_tag = args.model_tag or args.model.replace(".", "_").replace("-", "_")
    for domain in args.domains:
        generate_domain(args, client, model_tag, domain)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="gpt-5.6-sol")
    p.add_argument("--model-tag", default=None, help="file name tag; derived from --model if unset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())