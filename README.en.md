<p align="center">
  <img src="assets/logo.svg" alt="Zleap AI" width="220" />
</p>

# SAG Benchmark

> Companion benchmark reproduction repository for the SAG paper. This repository is for reproducing the paper benchmark scores with the quick-start commands, not for a general end-user product walkthrough.

English | [中文](README.md)

**Paper:** To be added

<p align="center">
  <img src="assets/sag-benchmark-simple.png" alt="SAG benchmark results" width="760" />
</p>

## Benchmark Score Reproduction

This repository contains upload, retrieval, and Recall evaluation scripts for SAG on HotpotQA, 2WikiMultiHopQA, and MuSiQue. The main goal is to let readers run the quick-start commands below and reproduce the benchmark scores reported in the paper, especially Recall@1 / Recall@2 / Recall@5 / Recall@10.

Default paper setup:

| Item | Value |
|------|------|
| Embedding | `bge-large-en-v1.5` |
| LLM | `qwen3.6-flash` |
| Main metrics | Recall@1 / Recall@2 / Recall@5 / Recall@10 |
| Main scripts | `scripts/run_upload.py`, `scripts/run_search_benchmark.py` |

Reference results:

| Method | Dataset | Recall@1 | Recall@2 | Recall@5 | Recall@10 |
|------|--------|----------|----------|----------|-----------|
| **SAG** | HotpotQA | **47.80%** | **91.55%** | **96.50%** | **97.70%** |
| HippoRAG 2 | HotpotQA | 44.40% | 78.35% | 94.35% | 97.15% |
| **SAG** | 2WikiMultiHopQA | **43.53%** | **82.30%** | 88.00% | 88.75% |
| HippoRAG 2 | 2WikiMultiHopQA | 42.38% | 76.55% | 90.35% | 93.40% |
| **SAG** | MuSiQue | **36.17%** | **64.05%** | **80.04%** | **83.37%** |
| HippoRAG 2 | MuSiQue | 30.65% | 49.52% | 65.13% | 73.76% |
| **SAG** | **Average** | **42.50%** | **79.30%** | **88.18%** | **89.94%** |
| HippoRAG 2 | **Average** | 39.14% | 68.14% | 83.28% | 88.10% |

With NV-Embed-v2, SAG reaches **81.71%** Recall@5 on MuSiQue, compared with 74.55% for HippoRAG 2.

## Method Figures

<p align="center">
  <img src="assets/paper-rag-comparison.png" alt="Naive RAG, GraphRAG and SAG comparison" width="760" />
</p>

SAG organizes text into lightweight `chunk -> event`, `chunk -> entities`, and `event <-> entities` indexes. It does not maintain a heavy global knowledge graph; it uses the event/entity index for SQL, vector search, full-text search, and multi-hop expansion.

<p align="center">
  <img src="assets/paper-sag-architecture.png" alt="SAG architecture" width="760" />
</p>

## Quick Start

### 1. Install Dependencies

Requirements:

- Python 3.11+
- `uv`
- Docker Compose
- Available LLM, embedding, and rerank endpoints

```bash
uv sync
cp .env.example .env
```

Edit `.env` and fill in MySQL, Elasticsearch, LLM, embedding, and rerank settings. Do not commit real secrets.

### 2. Start Local Services

All local services are managed by `docker-compose.yml`.

| Service | Container | Default port | Notes |
|------|--------|----------|------|
| MySQL | `sag2_mysql` | `3306` | Default user `sag2` |
| Elasticsearch | `new_sag_elasticsearch` | `9200` | Security disabled |
| MLflow | `sag2_mlflow` | `5000` | Optional experiment tracking |

Ports can be overridden in `.env` with `MYSQL_PORT`, `ES_PORT`, and `MLFLOW_PORT`.

```bash
docker compose up -d
docker compose ps
```

### 3. Initialize Database and Indexes

```bash
uv run python scripts/init_database.py --fix-grants
uv run python scripts/init_elasticsearch.py
```

### 4. Upload Datasets

`run_upload.py` first converts `pipeline/evaluation/dataset/<dataset>.json` into a Markdown corpus, then writes it to MySQL and Elasticsearch. After upload, it generates:

```text
pipeline/evaluation/source/SAG/<LLM_MODEL>/<dataset>/<timestamp>/source_info.json
```

The file contains the `source_config_id` used by benchmark runs.

```bash
uv run python scripts/run_upload.py --dataset hotpotqa
uv run python scripts/run_upload.py --dataset 2wikimultihopqa
uv run python scripts/run_upload.py --dataset musique
```

For a quick smoke test, use the smaller datasets first:

```bash
uv run python scripts/run_upload.py --dataset test_hotpotqa
uv run python scripts/run_upload.py --dataset sample
```

To reproduce the **triplet (atomic event)** mode — where each event contains exactly 2 entities (subject-relation-object) — add `--atomic` when uploading:

```bash
uv run python scripts/run_upload.py --dataset sample --atomic
```


### 5. Run Paper Reproduction Benchmarks

Quick validation:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name test_hotpotqa \
  --strategy multi \
  --top-k 10 \
  --k-values "1,2,5,10" \
  --max-concurrency 5 \
  --limit 10
```

Main datasets:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name hotpotqa \
  --strategy multi \
  --top-k 10 \
  --k-values "1,2,5,10" \
  --max-concurrency 10 \
  --bench-size 20

uv run python scripts/run_search_benchmark.py \
  --dataset-name 2wikimultihopqa \
  --strategy multi \
  --top-k 10 \
  --k-values "1,2,5,10" \
  --max-concurrency 10 \
  --bench-size 20

uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy multi \
  --top-k 10 \
  --k-values "1,2,5,10" \
  --max-concurrency 10 \
  --bench-size 20
```

To pin a specific uploaded source, pass the `source_config_id` generated during upload:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy multi \
  --source-config-id musique-20260512_213908 \
  --top-k 10 \
  --k-values "1,2,5,10" \
  --max-concurrency 10
```

Enable MLflow:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy multi \
  --use-mlflow \
  --mlflow-url http://localhost:5000 \
  --mlflow-experiment sag-benchmark
```

Default output directory:

```text
output/<dataset>/<strategy>/<timestamp>/
```

Main output files:

| File | Description |
|------|------|
| `search_results.json` | Per-question retrieval results |
| `benchmark_results.json` | Recall, Precision, F1, and summary metrics |
| `run.log` | Run log |

## Datasets

| Name | Description |
|------|------|
| `hotpotqa` | HotpotQA multi-hop QA |
| `2wikimultihopqa` | 2WikiMultiHopQA |
| `musique` | MuSiQue multi-hop QA |
| `test_hotpotqa` | Small HotpotQA test set |
| `sample` | Tiny sample set for pipeline debugging |

Dataset files are under `pipeline/evaluation/dataset/`.

## Retrieval Strategies

| Strategy | Description |
|------|------|
| `multi` | Multi-route retrieval with NER, entity vector recall, multi-hop expansion, and merged ranking |
| `multi1` | Fixed 1-hop expansion plus dynamic expansion until the candidate budget is reached |
| `multi_es` | Multi-route implementation with `--mode fast/precise` |
| `hopllm` | Coarse retrieval followed by seed-based hop expansion |
| `atomic` | Entity-first atomic retrieval with step-by-step hop expansion |
| `vector` | Pure vector retrieval baseline |

See [docs/search.md](docs/search.md) for full arguments.

## Common Scripts

### Search Only

```bash
uv run python scripts/run_search.py \
  --dataset-name test_hotpotqa \
  --strategy multi \
  --output-dir output/manual-search
```

### Evaluate Existing Results

```bash
uv run python scripts/run_benchmark.py \
  --results output/<dataset>/<strategy>/<timestamp>/search_results.json \
  --dataset musique
```

### Compare Two Retrieval Outputs

```bash
uv run python scripts/compare_recall_methods.py \
  --predictions \
    output/test_hotpotqa/multi/run_a/search_results.json \
    output/test_hotpotqa/vector/run_b/search_results.json \
  --dataset-name test_hotpotqa \
  --k-values 1,2,5,10 \
  --verbose
```

## Repository Layout

```text
SAG-Benchmark/
├── assets/                         # README figures and logo
├── pipeline/
│   ├── core/                       # Config, AI clients, storage layer
│   ├── db/                         # SQLAlchemy ORM
│   ├── evaluation/
│   │   ├── dataset/                # Evaluation datasets
│   │   ├── metrics/                # Recall and related metrics
│   │   └── utils/                  # Data loading, MLflow, token tracking
│   ├── modules/
│   │   ├── extract/                # Event/entity extraction
│   │   ├── load/                   # Document loading and chunking
│   │   └── search/                 # Retrieval strategies
│   └── utils/
├── scripts/
│   ├── init_database.py
│   ├── init_elasticsearch.py
│   ├── run_upload.py
│   ├── run_search_benchmark.py
│   ├── run_search.py
│   ├── run_benchmark.py
│   └── compare_recall_methods.py
├── docs/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Reproduction Notes

- Results depend on the actual LLM, embedding, and rerank services configured in `.env`; changing models, embedding dimensions, or rerank settings can change the metrics.
- When `--source-config-id` is omitted, `run_search_benchmark.py` looks up the latest uploaded source based on `LLM_MODEL` in `.env`.
- Full dataset upload and benchmark runs call external model services. Check quota, concurrency, and timeout settings before running.
- Stop local services with `docker compose down`. To delete local database volumes, use `docker compose down -v`; this removes uploaded data.
