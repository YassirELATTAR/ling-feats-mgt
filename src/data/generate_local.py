"""
Continuation generation with locally hosted models (Llama 3.1 / 3.3 / 4).
Prompt = raw 30-word prefix, no instruction, as in MAGE (Appendix A).

  python generate_local.py -m meta-llama/Llama-3.1-8B --data-root data --family llama
  python generate_local.py -m meta-llama/Llama-4-Scout-17B-16E --data-root data --family llama --load-4bit --batch-size 4

Reads  {data_root}/{split}/Human/{domain}/samples.csv
Writes {data_root}/{split}/AI/{domain}/{family}/{model_tag}.csv  (doc_id,text,label)
Resume-safe: finished files are skipped, interrupted ones continue from a checkpoint.
"""
import argparse
import os
import random

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOMAINS = ["cmv", "yelp", "xsum", "tldr", "eli5", "wp", "roct", "hswag", "squad", "sci_gen"]
SPLIT_SIZES = {"train": 800, "validation": 100, "test": 100}


def load_model(args):
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "cache_dir": args.cache_dir}
    if args.max_memory_per_gpu:
        kwargs["max_memory"] = {i: args.max_memory_per_gpu
                                for i in range(torch.cuda.device_count())}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    if "Llama-4" in args.model:
        from transformers import Llama4ForConditionalGeneration
        model_cls = Llama4ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(args.model, **kwargs).eval()
    print(f"device map: {model.hf_device_map}", flush=True)
    return tok, model


def generate_split(args, tok, model, model_tag, split, domain, n_sample):
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

    print(f"\n=== {split}/{domain}: {len(prefixes)} texts | {model_tag} ===", flush=True)
    for i in range(start, len(prefixes), args.batch_size):
        batch = prefixes[i:i + args.batch_size]
        try:
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_tokens,
                                     do_sample=True, temperature=args.temperature,
                                     top_p=0.95, pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pl.DataFrame(rows).write_csv(ckpt_path)
            raise SystemExit(f"OOM at sample {i}; checkpoint written, rerun with a smaller --batch-size")

        for j in range(len(batch)):
            cont = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            rows.append({"doc_id": f"{model_tag}_{i + j + 1}",
                         "text": batch[j] + " " + cont,
                         "label": 1})

        if len(rows) % (args.batch_size * args.save_every) < args.batch_size:
            pl.DataFrame(rows).write_csv(ckpt_path)
            print(f"{len(rows)}/{len(prefixes)}", flush=True)

    pl.DataFrame(rows).write_csv(out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"saved {len(rows)} -> {out_path}", flush=True)


def main(args):
    model_tag = args.model_tag or args.model.split("/")[-1].replace(".", "_").replace("-", "_")
    tok, model = load_model(args)
    for split, n_sample in SPLIT_SIZES.items():
        for domain in DOMAINS:
            generate_split(args, tok, model, model_tag, split, domain, n_sample)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--model-tag", default=None, help="folder/file name; derived from --model if unset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--family", default="llama", help="family folder in the output path")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--max-memory-per-gpu", default=None, help='e.g. "45GiB"')
    p.add_argument("--prefix-words", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10, help="checkpoint every N batches")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-4bit", action="store_true")
    main(p.parse_args())