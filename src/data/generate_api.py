"""
Continuation generation through the OpenRouter API (GLM-5.2, Llama-4-Scout, GPT-5.6-sol).
Chat endpoints cannot do raw completion, so the 30-word prefix is wrapped in a minimal
system instruction. This is a documented deviation from MAGE's raw-prefix prompting.

  export OPENROUTER_API_KEY=...
  python generate_api.py -m z-ai/glm-5.2 --family glm --model-tag glm_5_2 --data-root data

Reads  {data_root}/{split}/Human/{domain}/samples.csv
Writes {data_root}/{split}/AI/{domain}/{family}/{model_tag}.csv  (doc_id,text,label)
Resume-safe: finished files are skipped, interrupted ones continue from a checkpoint.
"""
import argparse
import os
import random
import time

import polars as pl
from openai import OpenAI

DOMAINS = ["cmv", "yelp", "xsum", "tldr", "eli5", "wp", "roct", "hswag", "squad", "sci_gen"]
SPLIT_SIZES = {"train": 800, "validation": 100, "test": 100}

SYSTEM_PROMPT = ("Continue the following text naturally in the same style. "
                 "Output only the continuation, with no explanation or preamble, "
                 "and do not repeat the given text.")


def generate(client, model, prefix, max_tokens, temperature, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prefix}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"reasoning": {"enabled": False}})
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def generate_split(args, client, model_tag, split, domain, n_sample):
    in_path = os.path.join(args.data_root, split, "Human", domain, "samples.csv")
    if not os.path.exists(in_path):
        print(f"MISSING: {in_path}")
        return

    out_dir = os.path.join(args.data_root, split, "AI", domain, args.family)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_tag}.csv")
    ckpt_path = os.path.join(out_dir, f"{model_tag}_checkpoint.csv")

    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        return

    texts = [str(t) for t in pl.read_csv(in_path)["text"].to_list()]
    rng = random.Random(args.seed)
    if len(texts) > n_sample:
        texts = rng.sample(texts, n_sample)
    prefixes = [" ".join(t.split()[:args.prefix_words]) for t in texts]

    rows, start = [], 0
    if os.path.exists(ckpt_path):
        rows = pl.read_csv(ckpt_path).to_dicts()
        start = len(rows)
        print(f"resuming {split}/{domain} at {start}/{len(prefixes)}")

    print(f"\n=== {split}/{domain}: {len(prefixes)} texts | {args.model} ===", flush=True)
    for i in range(start, len(prefixes)):
        cont = generate(client, args.model, prefixes[i], args.max_tokens, args.temperature)
        if cont is None:
            pl.DataFrame(rows).write_csv(ckpt_path)
            raise SystemExit(f"failed at sample {i + 1}; checkpoint written, rerun to resume")
        rows.append({"doc_id": f"{model_tag}_{i + 1}",
                     "text": prefixes[i] + " " + cont,
                     "label": 1})
        if (i + 1) % args.save_every == 0:
            pl.DataFrame(rows).write_csv(ckpt_path)
            print(f"{i + 1}/{len(prefixes)}", flush=True)

    pl.DataFrame(rows).write_csv(out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"saved {len(rows)} -> {out_path}", flush=True)


def main(args):
    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"])
    model_tag = args.model_tag or args.model.split("/")[-1].replace(".", "_").replace("-", "_")
    for split, n_sample in SPLIT_SIZES.items():
        for domain in DOMAINS:
            generate_split(args, client, model_tag, split, domain, n_sample)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--model-tag", default=None, help="folder/file name; derived from --model if unset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--family", required=True, help="family folder in the output path")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--prefix-words", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())