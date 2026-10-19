<p align="center"><img src="assets/logo.svg" alt="Zleap AI" width="220" /></p>

# SAG2

> Companion benchmark reproduction repository for the SAG2 paper, with code, data and instructions for reproducing its experiments. For the SAG2 application, see [SAG-Wiki](https://github.com/Zleap-AI/SAG-Wiki).

**English** | [简体中文](README-CN.md) | [Project home](../README.md)

**Paper:** [SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV (arXiv:2610.22588)](https://arxiv.org/abs/2610.22588)

**SAG2: embedding-free multi-hop evidence retrieval.** SQLite FTS5, BM25 and JEV follow **question → passages → terms → new passages** at query time. Indexing builds a local lexical index. Agentic SAG2 and local reranking are optional extensions.

| Method to reproduce | Documentation and code |
|---|---|
| **SAG2**: BM25, JEV and term expansion | This README; the sag2/ subproject |
| **Original SAG paper**: events, entities and graph retrieval | **[Original SAG reproduction guide](../sag/README.md)**; independent `sag/` subproject |

<video src="https://github.com/Zleap-AI/SAG-Benchmark/raw/refs/heads/main/sag2/assets/SAG2-English.mp4" controls width="100%"></video>

**Contents**

- **SAG2**
  - [Benchmark Score Reproduction](#benchmark-score-reproduction)
  - [Method Figures](#method-figures)
  - [Quick Start](#quick-start)
  - [Datasets](#datasets)
  - [Retrieval Configuration](#retrieval-configuration)
  - [Outputs and Resume](#outputs-and-resume)
  - [Common Scripts](#common-scripts)
- **Extensions**
  - [Agentic SAG2](#agentic-sag2)
  - [Local Reranking](#local-reranking)
  - [Run External Methods](../README.md#run-external-methods)
- **Code and Reproduction**
  - [Repository Layout](#repository-layout)
  - [Reproduction Notes and Ablations](#reproduction-notes-and-ablations)
  - [Original SAG Paper Readers](#original-sag-paper-readers)
  - [Indexing and Cost](#indexing-and-cost)
  - [Indexes and Evaluation Settings](#indexes-and-evaluation-settings)
  - [Offline Verification](#offline-verification)

## Benchmark Score Reproduction

SAG2 was evaluated on **1,000 questions each** from MuSiQue, 2WikiMultiHopQA and HotpotQA. R@10 is **86.62%, 97.43% and 99.00%**, respectively. EnterpriseRAG-Bench covers **511,962 source documents**, with document R@10 of **81.10%** over 470 labeled questions.

The paper's main table is below; see the [evaluation settings](#indexes-and-evaluation-settings) for its configuration. Values are percentages; bold denotes the column maximum. R@k is the mean fraction of labeled evidence recovered in the first k deduplicated results; AE@k is the fraction of questions with all evidence recovered. **R@k reports evidence recall.**

<table>
  <thead>
    <tr><th rowspan="2">Method</th><th colspan="4">MuSiQue</th><th colspan="4">2WikiMultiHopQA</th><th colspan="4">HotpotQA</th></tr>
    <tr><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th></tr>
  </thead>
  <tbody>
    <tr><th colspan="13">Lexical and dense retrieval</th></tr>
    <tr><td>BM25</td><td align="right">28.53</td><td align="right">37.32</td><td align="right">47.47</td><td align="right">54.41</td><td align="right">40.10</td><td align="right">57.93</td><td align="right">67.05</td><td align="right">71.50</td><td align="right">39.75</td><td align="right">57.95</td><td align="right">74.40</td><td align="right">86.90</td></tr>
    <tr><td>Contriever</td><td align="right">24.50</td><td align="right">33.54</td><td align="right">45.32</td><td align="right">53.58</td><td align="right">34.40</td><td align="right">48.60</td><td align="right">59.25</td><td align="right">65.20</td><td align="right">39.30</td><td align="right">57.45</td><td align="right">74.85</td><td align="right">82.75</td></tr>
    <tr><th colspan="13">Large embedding models</th></tr>
    <tr><td>BGE-Large-EN-v1.5</td><td align="right">29.42</td><td align="right">41.68</td><td align="right">56.32</td><td align="right">64.08</td><td align="right">41.48</td><td align="right">62.28</td><td align="right">69.83</td><td align="right">73.75</td><td align="right">45.45</td><td align="right">76.20</td><td align="right">89.20</td><td align="right">93.35</td></tr>
    <tr><td>GTE-Qwen2-7B</td><td align="right">35.19</td><td align="right">48.06</td><td align="right">61.92</td><td align="right">70.80</td><td align="right">43.45</td><td align="right">65.58</td><td align="right">74.18</td><td align="right">78.63</td><td align="right">46.40</td><td align="right">73.95</td><td align="right">88.45</td><td align="right">93.55</td></tr>
    <tr><td>GritLM-7B</td><td align="right">34.86</td><td align="right">50.70</td><td align="right">66.22</td><td align="right">75.19</td><td align="right">41.85</td><td align="right">66.23</td><td align="right">75.62</td><td align="right">79.40</td><td align="right">46.40</td><td align="right">81.15</td><td align="right">93.40</td><td align="right">97.20</td></tr>
    <tr><td>NV-Embed-v2</td><td align="right">34.02</td><td align="right">51.08</td><td align="right">68.84</td><td align="right">77.27</td><td align="right">43.23</td><td align="right">68.93</td><td align="right">77.00</td><td align="right">80.75</td><td align="right">46.90</td><td align="right">85.55</td><td align="right">95.25</td><td align="right">97.70</td></tr>
    <tr><th colspan="13">Structure-enhanced retrieval</th></tr>
    <tr><td>HippoRAG 2</td><td align="right">30.65</td><td align="right">49.52</td><td align="right">65.13</td><td align="right">73.76</td><td align="right">42.38</td><td align="right">76.55</td><td align="right">90.35</td><td align="right">93.40</td><td align="right">44.40</td><td align="right">78.35</td><td align="right">94.35</td><td align="right">97.15</td></tr>
    <tr><td>SAG (Ours)</td><td align="right"><strong>36.82</strong></td><td align="right"><strong>63.62</strong></td><td align="right">80.36</td><td align="right">83.37</td><td align="right"><strong>43.81</strong></td><td align="right">83.93</td><td align="right">93.34</td><td align="right">93.57</td><td align="right">47.80</td><td align="right">91.55</td><td align="right">96.50</td><td align="right">97.70</td></tr>
    <tr><td><strong>SAG2 (Ours)</strong></td><td align="right">34.30</td><td align="right">62.18</td><td align="right"><strong>80.68</strong></td><td align="right"><strong>86.62</strong></td><td align="right">43.55</td><td align="right"><strong>85.50</strong></td><td align="right"><strong>97.35</strong></td><td align="right"><strong>97.43</strong></td><td align="right"><strong>48.30</strong></td><td align="right"><strong>93.60</strong></td><td align="right"><strong>98.45</strong></td><td align="right"><strong>99.00</strong></td></tr>
  </tbody>
</table>

BM25 uses the paper's fixed lexical reference; cited baselines and provenance are in [main table data](results/paper/main-results.json). These are fixed development subsets; MuSiQue has 518 two-hop, 316 three-hop and 166 four-hop questions and was used for method development. Enterprise results are shown below. [Ablations](#reproduction-notes-and-ablations) and Agentic results are reported in later sections.

EnterpriseRAG-Bench contains 511,962 source documents. The table reports document R@10 (%) on 470 labeled questions.

| Question type | n | BM25 | Vector | Bash Agent† | **SAG2 (Ours)** |
| --- | --- | --- | --- | --- | --- |
| Basic retrieval | 175 | 77.7 | 48.6 | 61.7 | **86.86** |
| Semantic retrieval | 125 | 43.2 | 24.8 | 29.6 | **58.40** |
| Within-document reasoning | 40 | 90.0 | 57.5 | 75.0 | **100.00** |
| Project-related | 40 | 65.5 | 49.2 | 43.2 | **83.58** |
| Constrained | 30 | 85.0 | 83.3 | 78.3 | **96.67** |
| Conflicting information | 20 | 82.5 | 55.0 | 72.5 | **92.50** |
| Completeness | 20 | 46.5 | 33.4 | 59.0 | **76.24** |
| Miscellaneous | 20 | 90.0 | 75.0 | 100.0 | **100.00** |
| Overall | 470 | 68.4 | 46.0 | 55.8 | **81.10** |

Enterprise baselines are cited from the [dataset paper](https://arxiv.org/abs/2605.05253). Bash Agent has variable output size; other methods in this table return ten documents, so computation and output budgets differ.

## Method Figures

![SAG2 indexing and retrieval](assets/sag2-workflow.png)

1. **Index source text.** Tokenize titles and text into SQLite FTS5; retain source identity, headings and term statistics. No embeddings or offline semantic extraction are required.
2. **Retrieve passages from the question.** BM25 returns up to 100 candidates. JEV `noul` scores whether each passage supplies necessary evidence for the original question and available context.
3. **Select terms from passages.** Up to 10 expansion seeds contribute up to 20 terms, using frequency, rarity, query context and JEV scores.
4. **Retrieve more evidence.** Search each term, interleave and deduplicate results, retain earlier evidence and score again. The default adds two expansion rounds.
5. **Return sources.** Locally diversify the top ten results without another model request.

JSON corpus text is treated as plain text by default; a supplied `path` selects Markdown or code parsing by its file extension. No generated annotations are added to source text. Gold answers and evidence labels are used only for evaluation. The algorithm, prompts and query optimizations are in [scripts/sag2.py](scripts/sag2.py). The original event/entity approach is documented in [sag/](../sag/README.md).

## Quick Start

Run the following commands from **SAG-Benchmark/sag2/**. Original SAG readers should instead follow [sag/README.md](../sag/README.md), which has its own environment and dependencies.

### 1. Install

Requirements: Python **3.11+**, [uv](https://docs.astral.sh/uv/getting-started/installation/) and a C compiler. The SQLite scoring extension compiles on first use: Xcode Command Line Tools on macOS, GCC or Clang on Linux, or MSVC Build Tools on Windows. Default SAG2 needs no Docker, MySQL, Elasticsearch or GPU. Shell examples use Bash (Linux, macOS or WSL); in PowerShell, use Copy-Item for the configuration copy. Paper-control scripts currently require a Unix-like environment.

```bash
git clone https://github.com/Zleap-AI/SAG-Benchmark.git
cd SAG-Benchmark/sag2
uv sync --frozen
```

For an existing checkout, run `cd sag2` from the repository root. Run subsequent SAG2 commands here, using `sag2/.venv/`.

### 2. Configure JEV

Create the configuration once, preserving an existing file:

```bash
test -f .env || cp .env.example .env
```

Set these values in the current directory’s `.env` (repository path: `sag2/.env`):

```dotenv
OPENROUTER_API_KEY=your_api_key
JEV_BASE_URL=https://openrouter.ai/api/alpha/decisions
JEV_MODEL=typesafe/jev-1.13
```

The default answer type is `noul`. The historical paper run returned `typesafe/jev-1.13-20260917`; provider versions may change, so record the actual returned model. Keys are read from environment variables or `--env-file`, never passed as command-line arguments.

### 3. Prepare Data and Indexes

```bash
# Download missing pinned data and index all three datasets; no model requests
uv run python scripts/run_sag2_benchmark.py --download --index-only
```

An index with the same corpus and chunking configuration is reused. The terminal prints source count, chunk count and question count; the suite summary reports `indexed`. Indexes default to `.cache/sag2/`. Add `--dataset musique` to prepare only one dataset.

### 4. Run Ten Questions First

```bash
uv run python scripts/run_sag2_benchmark.py \
  --dataset musique --limit 10 --concurrency 20 \
  --output-dir output/sag2/musique-smoke
```

`--limit` changes the question count, not the corpus size. This step uses the Jev API. The terminal prints recall metrics; per-question results are in `output/sag2/musique-smoke/musique/details.jsonl`, with `metrics.json` in the same directory.

### 5. Run the Full Benchmarks

```bash
# 1,000 questions per dataset
uv run python scripts/run_sag2_benchmark.py --datasets all --concurrency 20

# Or choose specific datasets
uv run python scripts/run_sag2_benchmark.py \
  --datasets hotpotqa 2wikimultihopqa --concurrency 20
```

Add `--output-dir output/sag2/my-run` for an explicit destination; it must not already exist for a new run. Check dependencies, data and the small run before starting full evaluation.

## Datasets

| Dataset argument | Questions | Source passages / documents | Evaluation unit |
|---|---:|---:|---|
| `musique` | 1,000 | 11,656 | Supporting passage |
| `2wikimultihopqa` | 1,000 | 6,119 | Supporting passage |
| `hotpotqa` | 1,000 | 9,811 | Supporting passage |
| EnterpriseRAG-Bench, separate entry point | 470 labeled | 511,962 | Source document |

The three multi-hop datasets come from HippoRAG 2 at revision `5ec05b38deecc3318bb432c69865959c56058990`. `--datasets all` means the first three rows. Enterprise uses the separate entry point below. See [dataset/README.md](dataset/README.md) for format and license information.

### EnterpriseRAG-Bench

Prepare the official directory with `documents/`, `questions.jsonl` and `download_manifest.json`. **Check for an existing matching index before building another one.**

```bash
# Only for the first build of a current-format index; preserve older indexes
uv run python scripts/build_enterprise_index.py \
  --dataset-dir ../../datasets/EnterpriseRAG-Bench \
  --index-path .cache/sag2/enterprise.sqlite \
  --output-dir output/enterprise/index

# Reuse that index and evaluate all 470 labeled questions
uv run python scripts/run_enterprise.py \
  --dataset-dir ../../datasets/EnterpriseRAG-Bench \
  --index-summary output/enterprise/index/summary.json \
  --output-dir output/enterprise/evaluation --concurrency 20
```

The evaluator opens the index read-only and writes `progress.json`, `summary.json`, per-question `qst_*.json` files and `requests/`. Repeating the same command reuses saved results; use a new output directory when changing models, data or settings. A schema4 index cannot be passed directly to the current schema5 reader. Keep the old file and use its matching compatibility implementation; changing a version marker does not convert an index.

## Retrieval Configuration

| Argument | Default | Purpose |
|---|---|---|
| `--mode` | `sag2` | `sag2`, `agentic` or `local` |
| `--dataset` / `--datasets` | `all` | One or more multi-hop datasets |
| `--dataset-dir` | `dataset/` | Dataset files; defaults to `sag2/dataset/` |
| `--limit` / `--offset` | All / `0` | Question range; corpus stays complete |
| `--concurrency` | `20` | Concurrent questions |
| `--index-workers` | `8` | Index-building processes, separate from query concurrency |
| `--candidate-k` | `100` | Lexical candidates per round |
| `--seed-k` | `10` | Expansion seeds per round |
| `--expansions` / `--max-hops` | `2` for SAG2; `1` for Agentic | Additional rounds after the initial retrieval; 0, 1 or 2 |
| `--expansion-term-limit` | `20` | Expansion terms per round |
| `--top-k` / `--k-list` | `10` / `1 2 5 10` | Output size and evaluation cutoffs |
| `--chunk-chars` | `2200` | Chunk target in UTF-16 units |
| `--cache-dir` | `.cache/sag2` | SQLite index directory |
| `--output-dir` | Timestamped directory | Destination for a new run |
| `--resume` | Unset | Continue an existing run with its original settings |
| `--env-file` | `sag2/.env` (repository path) | Configuration file; `SAG_ENV_FILE` can override the default |

```bash
# One expansion round, still using 100 candidates and 10 seeds
uv run python scripts/run_sag2_benchmark.py \
  --dataset musique --limit 10 --expansions 1 --candidate-k 100 --seed-k 10

uv run python scripts/run_sag2_benchmark.py --help
```

Changing models, chunking or retrieval parameters changes the experiment. Match the model version, corpus, questions, chunking policy and metric units when comparing with published scores.

## Outputs and Resume

The following paths are relative to `SAG-Benchmark/sag2/`. Default output layout:

```text
output/sag2/<timestamp>/
├── summary.json
└── <dataset>/
    ├── config.json
    ├── details.jsonl
    ├── metrics.json
    ├── search_results.json
    ├── bm25_results.json
    └── requests/          # Agentic / local reranking modes
```

| File | Content and completion checks |
|---|---|
| `summary.json` | Final per-dataset summary; index-only runs report `indexed` |
| `config.json` | Model, settings, data and code hashes |
| `details.jsonl` | One completed question per line, with retrieval, metrics and usage |
| `metrics.json` | Final dataset metrics, all-evidence recall and usage |
| `search_results.json` | Final retrieved source text per question, for QA or analysis |
| `bm25_results.json` | Initial BM25 results for the same questions |
| `requests/` | Agentic and local request/response records for recovery |

```bash
# Reuse the original datasets, range, model and settings
uv run python scripts/run_sag2_benchmark.py \
  --dataset musique --limit 10 --concurrency 20 \
  --resume output/sag2/musique-smoke
```

Plain SAG2 resumes completed questions. Agentic and local modes also reuse successful identical requests. Plain SAG2 does not persist each Jev request separately for recovery, so an interrupted question may send its requests again. Keep different model/configuration runs in separate output directories.

## Common Scripts

| Script | Purpose |
|---|---|
| `scripts/run_sag2_benchmark.py` | Multi-hop indexing, SAG2, Agentic and local evaluation |
| `scripts/build_enterprise_index.py` | Full Enterprise index construction |
| `scripts/run_enterprise.py` | Enterprise evaluation over an existing read-only index |
| `scripts/run_paper_controls.py` | MuSiQue paper controls, including model calls |
| `scripts/verify_paper_results.py` | Offline verification of retained paper records |
| `scripts/verify_agentic_results.py` | Verify 3,000 retained best Agentic results |
| `scripts/verify_rerank_results.py` | Verify retained reranking results |
| `scripts/search_local.py` | Query a custom JSON corpus with local reranking |

The current SAG2 entry points return evidence. The original SAG upload, answer-generation and LLM Judge workflows are preserved separately in [sag/](../sag/README.md).

## Agentic SAG2

**Optional extension.** An LLM retains the original question and retrieval history, calls SAG2 as a tool, decides when to search or stop, and ranks the final ten passages. It adds no generated text at index time.

Best historical results over 1,000 questions per dataset are below. MuSiQue reaches **92.20% R@5**, a gain of 11.53 percentage points over a single SAG2 run using unrounded values, and **95.39% R@10**. R@5 is 99.88% on 2WikiMultiHopQA and 99.30% on HotpotQA.

<table>
  <thead>
    <tr><th rowspan="2">Method</th><th colspan="4">MuSiQue</th><th colspan="4">2WikiMultiHopQA</th><th colspan="4">HotpotQA</th></tr>
    <tr><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th></tr>
  </thead>
  <tbody>
    <tr><td>SAG2</td><td align="right">34.30</td><td align="right">62.18</td><td align="right">80.68</td><td align="right">86.62</td><td align="right">43.55</td><td align="right">85.50</td><td align="right">97.35</td><td align="right">97.43</td><td align="right">48.30</td><td align="right">93.60</td><td align="right">98.45</td><td align="right">99.00</td></tr>
    <tr><td><strong>Agentic SAG2</strong></td><td align="right"><strong>38.21</strong></td><td align="right"><strong>71.99</strong></td><td align="right"><strong>92.20</strong></td><td align="right"><strong>95.39</strong></td><td align="right"><strong>44.08</strong></td><td align="right"><strong>88.00</strong></td><td align="right"><strong>99.88</strong></td><td align="right"><strong>99.88</strong></td><td align="right"><strong>49.10</strong></td><td align="right"><strong>96.45</strong></td><td align="right"><strong>99.30</strong></td><td align="right"><strong>99.65</strong></td></tr>
  </tbody>
</table>

Evaluation settings: `gpt-6.1-sol`, `high` reasoning, one expansion per SAG2 call, at most three tool calls per question, 20 passages returned per call and 10 final passages. The general API entry point requires no Codex account or CLI. Historical scores came from native MCP runs.

```dotenv
LLM_API_KEY=your_llm_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-6.1-sol
LLM_API=responses
LLM_REASONING_EFFORT=high
```

```bash
uv run python scripts/run_sag2_benchmark.py --mode agentic \
  --dataset musique --limit 10 --expansions 1 --max-tool-calls 3 --return-k 20
```

Configure `--llm-model`, `--llm-base-url`, `--llm-api responses|chat`, `--llm-api-key-env` and `--reasoning-effort`. The provider must support the selected tool-calling format. See the [Agentic guide](docs/agentic.md) and [best results](results/agentic/README.md) for configuration, tokens and costs.

## Local Reranking

**Optional extension.** Replace each Jev scoring round with a local model while keeping BM25, term expansion and final diversity ordering. After preparing weights and data, queries can run offline.

The following results were evaluated through 302 APIs, with 1,000 questions per dataset:

<table>
  <thead>
    <tr><th rowspan="2">Method</th><th colspan="4">MuSiQue</th><th colspan="4">2WikiMultiHopQA</th><th colspan="4">HotpotQA</th></tr>
    <tr><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@2</th><th>R@5</th><th>R@10</th></tr>
  </thead>
  <tbody>
    <tr><td>BM25</td><td align="right">28.71</td><td align="right">37.50</td><td align="right">47.66</td><td align="right">54.50</td><td align="right">40.25</td><td align="right">58.18</td><td align="right">67.08</td><td align="right">71.58</td><td align="right">39.75</td><td align="right">58.05</td><td align="right">74.45</td><td align="right">86.95</td></tr>
    <tr><td>SAG2 + Jev（paired control）</td><td align="right">34.24</td><td align="right">62.34</td><td align="right">80.59</td><td align="right">86.61</td><td align="right">43.55</td><td align="right">85.50</td><td align="right">97.35</td><td align="right">97.43</td><td align="right">48.30</td><td align="right">93.60</td><td align="right">98.45</td><td align="right">99.00</td></tr>
    <tr><td>SAG2 + Jina（two expansions）</td><td align="right">31.03</td><td align="right">50.38</td><td align="right">65.61</td><td align="right">72.13</td><td align="right">43.10</td><td align="right">70.15</td><td align="right">80.80</td><td align="right">86.95</td><td align="right">46.75</td><td align="right">85.55</td><td align="right">94.35</td><td align="right">97.00</td></tr>
  </tbody>
</table>

<details>
<summary>MuSiQue model and pipeline ablations</summary>

| MuSiQue ablation | R@1 | R@2 | R@5 | R@10 | AE@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 28.71 | 37.50 | 47.66 | 54.50 | 17.50 |
| BM25＋Qwen single scoring round | 31.66 | 45.44 | 60.17 | 66.88 | 32.50 |
| BM25＋Jina single scoring round | 35.01 | 52.18 | 63.04 | 67.43 | 34.70 |
| SAG2＋Qwen two expansions | 22.64 | 34.75 | 55.55 | 71.61 | 44.60 |
| SAG2＋Qwen 0.6B two expansions | 25.67 | 40.81 | 62.54 | 74.18 | 48.40 |
| SAG2＋Jina two expansions | 31.03 | 50.38 | 65.61 | 72.13 | 43.80 |
| SAG2＋Jev baseline | 34.24 | 62.34 | 80.59 | 86.61 | 69.00 |

Single-round runs use the original query and standard relevance instructions. Two-expansion runs use SAG2 evidence instructions, context and final DPP, so these comparisons change both pipeline and prompts. `qwen3-rerank` and Qwen 0.6B are distinct models.
</details>

```bash
# Prepare dependencies and pinned weights while online
uv sync --frozen --group local
uv run --group local hf download Qwen/Qwen3-Reranker-0.6B \
  --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 --local-dir models/qwen3-reranker-0.6b

# Run ten questions offline once data and weights are ready
uv run --offline --group local python scripts/run_sag2_benchmark.py \
  --mode local --reranker qwen --model-path models/qwen3-reranker-0.6b \
  --dataset musique --limit 10 --expansions 2 --concurrency 20
```

Available backends include Qwen Transformers, Jina Transformers and Jina MLX on Apple Silicon. Qwen weights use Apache-2.0; Jina uses CC BY-NC 4.0 and requires additional permission for commercial use. See the [local reranking guide](docs/local-rerank.md) for custom corpora, memory, batch sizes and full ablations.

## Repository Layout

```text
SAG-Benchmark/
├── README.md / README-CN.md   # SAG2 introduction, results and reproduction / SAG2介绍、成绩与复现
├── sag/                      # Original SAG / 原SAG
│   ├── README.md / README-CN.md
│   ├── pipeline/ / scripts/ / prompts/
│   ├── external/             # GraphRAG, HippoRAG 2, HyperGraphRAG, Hyper-RAG, LightRAG
│   └── pyproject.toml / uv.lock / docker-compose.yml
└── sag2/                     # SAG2 and its optional extensions
    ├── README.md / README-CN.md
    ├── pyproject.toml / uv.lock / .env.example
    ├── scripts/
    │   ├── run_sag2_benchmark.py
    │   ├── sag2.py / agentic_sag2.py / local_reranker.py
    │   └── build_enterprise_index.py / run_enterprise.py
    ├── prompts/ / docs/ / dataset/ / results/ / tests/ / assets/
    ├── .cache/sag2/          # SQLite indexes / SQLite索引（运行时生成）
    └── output/              # Evaluation results / 评测结果（运行时生成）
```

Each method has separate `.env`, `.venv`, `dataset/` and `output/` paths. Run SAG2 from `sag2/` and original SAG from `sag/`. Original SAG uses `--strategy sag` and the `SAGGraphSearcher` event/entity graph implementation; SQLite SAG2 is implemented separately in `sag2/`.

## Reproduction Notes and Ablations

### Original SAG Paper Readers

Original SAG stores passages, events and entities, links evidence through shared entities, and combines SQL, dense retrieval and LLM selection. Its paper implementation uses MySQL/Elasticsearch, BGE-Large-EN-v1.5 and Qwen3.6-Flash. SAG2 keeps query-time evidence expansion while using a lexical index and JEV scoring.

Start with [sag/README.md](../sag/README.md): install, initialize storage, upload, retrieve, generate answers, run Judge and reproduce ablations. The original code and documentation retain the BGE-Large-EN-v1.5 / Qwen3.6-Flash configuration. SAG2 Jev settings do not replace it. Check legacy Docker container names, ports and volumes if services already exist.

### Indexing and Cost

| MuSiQue index and cost | Paper record |
|---|---:|
| Source passages / indexed units | 11,656 / 11,656 |
| Index construction | 2.36 seconds |
| SQLite file | 25.87 MiB |
| Offline model calls / cost | 0 / USD 0 |
| Model cost / 100 questions | USD 0.433769 |
| Model cost / 1,000 questions | USD 4.326379 |

Construction time is the median of three builds on Apple M4 Pro, 64 GiB, eight processes. It includes tokenization, index writes, statistics and query initialization; it excludes imports, corpus loading and validation, and does not clear the OS file cache. Costs come from historical responses and exclude compute rental and answer generation. See [paper data](results/paper/).

<details>
<summary>Full MuSiQue controlled experiments, 1,000 questions each</summary>

| Method | R@1 | R@2 | R@5 | R@10 | AE@5 | AE@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 28.53 | 37.32 | 47.47 | 54.41 | 12.20 | 17.40 |
| BM25+JEV (H=0) | 31.13 | 46.79 | 60.78 | 67.93 | 26.60 | 36.00 |
| SAG2 (Ours), H=1 | 34.35 | 61.44 | 77.89 | 83.36 | 54.60 | 63.10 |
| SAG2 (Ours), H=2 | 34.30 | 62.18 | 80.68 | 86.62 | 59.00 | 69.50 |
| BM25+JEV，matched budget | 30.64 | 46.58 | 61.57 | 69.82 | 29.10 | 39.60 |
| SAG2 (Ours)，without context | 31.20 | 48.04 | 66.32 | 76.25 | 37.20 | 52.20 |
| Lexical expansion, final JEV only | 31.23 | 45.43 | 57.17 | 61.34 | 25.50 | 29.60 |
| RM3 | 21.79 | 28.50 | 39.69 | 47.83 | 9.80 | 15.70 |
| RM3+final JEV | 30.88 | 45.76 | 60.19 | 66.93 | 29.80 | 37.20 |
| GAR-BM25 | 28.71 | 37.50 | 47.66 | 54.50 | 12.10 | 17.50 |
| GAR-BM25+final JEV | 30.91 | 48.61 | 64.63 | 73.11 | 34.70 | 46.10 |

H counts expansion rounds after initial retrieval; H0/H1/H2 come from the same run. The matched-budget control matches passage judgments, not tokens or cost. [Control data](results/paper/controlled-study.json) · [Analysis](results/paper/controlled-analysis.json)
</details>

### Indexes and Evaluation Settings

Current chunking follows SAG-Desktop: a target of 2,200 UTF-16 units, complete sentences, heading paths and roughly 80 characters of whole-sentence overlap within a heading. Indexes store source ranges and no LLM annotations. Query optimizations include scoring-statistics caching, bounded candidate selection, parallel expansion and persisted corpus statistics.

The index signature includes schema and chunking policy. Matching configurations reuse indexes; changing chunking changes candidate boundaries and scores. Older Enterprise formats require their matching reader; do not overwrite an old index to handle compatibility.

### Offline Verification

Run these commands from **SAG-Benchmark/sag2/**.

```bash
uv run python scripts/verify_paper_results.py
uv run python scripts/verify_agentic_results.py
uv run python scripts/verify_rerank_results.py
uv run pytest -q
uv run ruff check scripts tests
```

HTTP 499 retries continue within the current request; temporary failures such as 500 and timeouts have bounded retries. See each mode's guide and implementation. Check the host, endpoint, concurrency and output directory before running.
