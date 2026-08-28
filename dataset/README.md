# Dataset provenance and licensing

The full benchmark datasets are **not distributed through this repository**.
They are third-party data, not authored by this repository; their copyright and
license terms remain with their upstream publishers. The repository's MIT
license covers first-party content (source code, scripts, prompts, docs, and
assets) but does **not** cover the contents of `dataset/`.

This directory ships only two small entries used by the quick-start smoke
tests:

- `sample.json` / `sample_corpus.json` — hand-authored tiny sample.
- `test_hotpotqa.json` / `test_hotpotqa_corpus.json` — HotpotQA subset.

## Downloading the full benchmarks

Download the remaining benchmark files
(`hotpotqa*.json`, `musique*.json`, `2wikimultihopqa*.json`,
`narrativeqa*.json`) from the
[OSU NLP Group's HippoRAG 2 data release](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
on Hugging Face and place them here:

```text
dataset/
├── hotpotqa.json              # from HippoRAG 2 release
├── hotpotqa_corpus.json
├── musique.json
├── musique_corpus.json
├── 2wikimultihopqa.json
├── 2wikimultihopqa_corpus.json
├── narrativeqa.json
├── narrativeqa_corpus.json
├── sample.json                # included in this repository
├── sample_corpus.json
├── test_hotpotqa.json         # included in this repository
└── test_hotpotqa_corpus.json
```

The files there are used without changes to the file contents, questions,
answers, corpora, or dataset splits, so scores stay comparable with the paper.
After downloading, `uv run python scripts/run_upload.py --dataset hotpotqa`
works as documented in the README.

By downloading these files you accept the upstream terms. The table below
records the original provenance and license of each benchmark. Users should
cite both the HippoRAG 2 data release and the original publication for every
benchmark used in an experiment.

| File(s) | Dataset | Upstream | License | Notes |
| --- | --- | --- | --- | --- |
| `hotpotqa.json`, `hotpotqa_corpus.json` | HotpotQA | https://hotpotqa.github.io | CC BY-SA 4.0 | ShareAlike terms apply to redistributions. |
| `musique.json`, `musique_corpus.json` | MuSiQue | https://github.com/stonybrooknlp/musique | CC BY 4.0 | |
| `2wikimultihopqa.json`, `2wikimultihopqa_corpus.json` | 2WikiMultiHopQA | https://github.com/Alab-NII/2wikimultihop | Apache-2.0 | Cite the COLING 2020 paper listed by the upstream project. |
| `narrativeqa.json`, `narrativeqa_corpus.json` | NarrativeQA | https://github.com/google-deepmind/narrativeqa | Annotations: Apache-2.0; **narrative text: third-party copyrighted works** (Project Gutenberg books + IMSDb film scripts), not redistributed by upstream | Both files inline the underlying documents, which carry their own copyright. Do not redistribute them. |
| `sample.json`, `sample_corpus.json` | (hand-authored sample) | — | Derived from a Wikipedia-style entry (CC BY-SA 4.0 applies). | |
| `test_hotpotqa.json`, `test_hotpotqa_corpus.json` | HotpotQA (subset) | https://hotpotqa.github.io | CC BY-SA 4.0 | |

## If you add or replace a dataset file

Update this table and re-check whether the new terms are compatible with
redistribution in this repository. Keep large or third-party-licensed files out
of git; they stay local only.
