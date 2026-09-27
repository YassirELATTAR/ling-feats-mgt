# Data

This folder is empty in the repository — the corpus is released separately.
Download it and unpack it here, keeping the structure below.

**Download:** [Zenodo record](10.5281/zenodo.23003160)

## Layout

```
data/
├── train/
│   ├── Human/{domain}/samples.csv
│   └── AI/{domain}/{family}/{model_tag}.csv
├── validation/          # same structure
└── test/
    ├── Human/{domain}/samples.csv
    ├── AI/{domain}/{family}/{model_tag}.csv
    └── unseen/
        ├── Human/{domain}/{domain}_human.csv
        └── AI/{domain}/{domain}_{model}.csv
```

Every CSV has the columns `doc_id, text, label` (`label`: 0 = human, 1 = AI).
Note that MAGE's own release uses the opposite convention (0 = machine);
`src/data/merge_new_models.py` flips the labels when building the files used by
the MAGE baselines.

Text domains: `cmv, yelp, xsum, tldr, eli5, wp, roct, hswag, squad, sci_gen`.
Unseen test domains: `cnn, dialogsum, imdb, pubmed`.

## What is in the release

The human texts and the generations from the 27 original MAGE models come from
[MAGE](https://huggingface.co/datasets/yaful/MAGE), restricted to
**continuation prompts only**. On top of that, the release contains our own
generations:

| Family | Models added | Split coverage |
|---|---|---|
| llama | Llama-3.1-8B, Llama-4-Scout-17B-16E | train / validation / test |
| glm | GLM-5.2 | train / validation / test |
| — | GPT-5.6-sol | unseen test domains only |

800 train / 100 validation / 100 test samples per domain and model, prompted
with the first 30 words of the corresponding human text.

## Regenerating (optional)

The two scripts in `src/data/` are provided **for reference**: the released data
already contains all generated samples, so nothing here needs to be run to
reproduce our results. Use them only to extend the corpus with further models.

```bash
# locally hosted models
python src/data/generate_local.py -m meta-llama/Llama-3.1-8B --family llama --data-root data

# API models (OpenRouter)
export OPENROUTER_API_KEY=...
python src/data/generate_api.py -m z-ai/glm-5.2 --family glm --model-tag glm_5_2 --data-root data
```

Both scripts skip files that already exist and resume interrupted runs from a
checkpoint, so they can be stopped and restarted safely.

**Prompting deviation.** MAGE prompts base models with the raw 30-word prefix
and no instruction. `generate_local.py` does the same. `generate_api.py` cannot:
chat endpoints do not expose raw completion, so the prefix is wrapped in a
minimal system instruction (see the script). Regenerated samples will therefore
not be identical to the released ones, since generation is sampled rather than
greedy.


## Licence and attribution

Human texts and the generations from the 27 original models are derived from
the MAGE dataset (Li et al., 2024) and remain subject to its terms; see
https://github.com/yafuly/MAGE. The generations we contribute (Llama-3.1-8B,
Llama-4-Scout-17B-16E, GLM-5.2, GPT-5.6-sol) are released under the same terms
for consistency. The underlying source corpora (IMDb, Yelp, CNN/DailyMail,
PubMed and others) carry their own licences, which are unaffected by this
release.

If you use this data, cite both the MAGE paper and ours.