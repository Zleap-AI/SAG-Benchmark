<p align="center">
  <img src="assets/logo.svg" alt="Zleap AI" width="220" />
</p>

# SAG Benchmark

> Companion benchmark reproduction repository for the SAG paper. This repository is for reproducing the paper benchmark scores with the quick-start commands. General users, please see the [SAG project](https://github.com/Zleap-AI/SAG).

English | [中文](README-CN.md)

**Paper:** [https://arxiv.org/abs/2608.12129](https://arxiv.org/abs/2608.12129)

https://github.com/user-attachments/assets/ac805e3c-ab52-4857-bef6-2865f3831b2f


## Benchmark Score Reproduction

This repository contains upload, retrieval, and evaluation scripts for SAG on HotpotQA, 2WikiMultiHopQA, and MuSiQue. The main goal is to let readers run the quick-start commands below, reproduce the benchmark results reported in the paper, and compare retrieval and RAG methods across retrieval, QA performance, and LLM-as-a-Judge evaluations.

Default paper setup:

| Item | Value |
|------|------|
| Embedding | `bge-large-en-v1.5` |
| LLM | `qwen3.6-flash` |
| Main paper metrics | Recall@5 / F1 |
| Main scripts | `scripts/run_upload.py`, `scripts/run_search_benchmark.py` |

Main results:

<p align="center">
  <img src="assets/main-result.png" alt="SAG main benchmark results" width="760" />
</p>

Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue show that SAG achieves the best retrieval and end-to-end QA performance on every benchmark.

- Across the three datasets, SAG averages **90.07%/72.96%** in Recall@5 and F1, outperforming the strongest baseline for each metric by **6.79/4.33** percentage points, respectively.
- On the most challenging MuSiQue dataset, SAG outperforms the strongest baseline for each metric by **11.52/7.01** percentage points in Recall@5 and F1, respectively.

> **Note:** The current code can reproduce the SAG results but does not yet support running the other comparison methods. Support for these methods will be added once the relevant code is ready.

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

Activate the virtual environment when you want to run Python commands directly:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Edit `.env` and fill in the storage backend, LLM, embedding, and rerank settings. Do not commit real secrets.

At minimum, check these `.env` values before running upload or search:

```env
STORAGE_PROFILE=mysql_es

LLM_API_KEY=sk-...
LLM_BASE_URL=https://...
LLM_MODEL=qwen3.6-flash
LLM_LANGUAGE=en

EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=http://...
EMBEDDING_MODEL_NAME=text-embedding-bge-large-en-v1.5
EMBEDDING_DIMENSIONS=1024

RERANK_BASE_URL=http://...
RERANK_MODEL_NAME=Qwen/Qwen3-Reranker-8B
RERANK_ENDPOINT=/rerank
```

Then fill the storage connection used by your profile:

```env
# mysql_es
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=sag2
MYSQL_PASSWORD=sag2_pass
MYSQL_DATABASE=sag2

# oceanbase_es / oceanbase_full
OCEANBASE_HOST=localhost
OCEANBASE_PORT=2881
OCEANBASE_USER=sag2@sag2
OCEANBASE_PASSWORD=sag2_pass
OCEANBASE_DATABASE=sag2

# mysql_es / oceanbase_es
ES_HOST=localhost
ES_PORT=9200
ES_SCHEME=http
```

Storage is selected through `STORAGE_PROFILE`. The application code uses one storage facade, so upload and search callers do not need to know whether vectors are stored in Elasticsearch or OceanBase.

| Profile | SQL database | Vector/search backend | Notes |
|------|------|------|------|
| `mysql_es` | MySQL | Elasticsearch | Default paper-compatible local setup |
| `oceanbase_es` | OceanBase | Elasticsearch | Uses OceanBase for structured tables, keeps ES for vector search |
| `oceanbase_full` | OceanBase | OceanBase | Stores vectors in OceanBase table columns and searches through OceanBase vector indexes |

`DATABASE_BACKEND` and `VECTOR_BACKEND` are advanced overrides. Leave them empty unless you need to bypass the profile mapping.

### 2. Start Local Services

All local services are managed by `docker-compose.yml`.

| Service | Container | Default port | Notes |
|------|--------|----------|------|
| MySQL | `sag2_mysql` | `3306` | Default user `sag2` |
| Elasticsearch | `new_sag_elasticsearch` | `9200` | Security disabled |
| OceanBase | `oceanbase-ce` | `2881` | Optional backend for `oceanbase_es` / `oceanbase_full` |
| MLflow | `sag2_mlflow` | `5000` | Optional experiment tracking |

Ports can be overridden in `.env` with `MYSQL_PORT`, `ES_PORT`, and `MLFLOW_PORT`. OceanBase is exposed on `2881` for SQL client traffic.

Choose one startup path for the selected `STORAGE_PROFILE`. Do not run all of them.

#### 2.1 `mysql_es`

```bash
docker compose up -d mysql elasticsearch
docker compose ps
```

#### 2.2 `oceanbase_es`

```bash
docker compose up -d oceanbase elasticsearch
docker compose ps
```

For `oceanbase_es`, wait until the OceanBase container log shows that tenant DDL is ready and `init.sql` has completed before running project initialization:

```text
==> sag2 tenant ready.
==> Waiting for sag2 tenant DDL...
==> sag2 tenant DDL ready.
==> init.sql executed.
==> All done.
```

#### 2.3 `oceanbase_full`

```bash
docker compose up -d oceanbase
docker compose ps
```

For `oceanbase_full`, also wait until the OceanBase container log shows that tenant DDL is ready and `init.sql` has completed:

```text
==> sag2 tenant ready.
==> Waiting for sag2 tenant DDL...
==> sag2 tenant DDL ready.
==> init.sql executed.
==> All done.
```

Optional MLflow tracking can be started separately:

```bash
docker compose up -d mlflow
```

### 3. Initialize Database and Indexes

Choose one initialization path for the selected `STORAGE_PROFILE`. Do not run all of them.

#### 3.1 `mysql_es`

```bash
uv run python scripts/init_database.py --fix-grants
uv run python scripts/init_elasticsearch.py
```

#### 3.2 `oceanbase_es`

```bash
uv run python scripts/init_database.py
uv run python scripts/init_elasticsearch.py
```

#### 3.3 `oceanbase_full`

```bash
uv run python scripts/init_database.py
```

`scripts/init_database.py` reads `STORAGE_PROFILE` and initializes the active SQL backend:

- `mysql_es`: creates the normal MySQL structured tables.
- `oceanbase_es`: creates the normal OceanBase structured tables and adds OceanBase-only compatibility columns such as `source_event.entities`.
- `oceanbase_full`: does everything from `oceanbase_es`, then idempotently adds OceanBase vector columns and vector indexes for chunks, events, entities, and event-entity relations.

Run `scripts/init_elasticsearch.py` only when the active vector backend is Elasticsearch (`mysql_es` or `oceanbase_es`). It is not required for `oceanbase_full`. The `--fix-grants` option is only for local MySQL permission repair; do not use it for OceanBase profiles.

### 4. Upload Datasets

`run_upload.py` first converts `pipeline/evaluation/dataset/<dataset>.json` into a Markdown corpus, then writes structured rows and vectors through the configured storage facade. With `mysql_es`, data goes to MySQL plus Elasticsearch. With `oceanbase_full`, structured data and vectors are written into OceanBase. After upload, it generates:

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
│   ├── storage/                    # Storage facade and backend providers
│   └── utils/
├── scripts/
│   ├── init_database.py
│   ├── init_oceanbase_vectors.py
│   ├── init_elasticsearch.py
│   ├── run_upload.py
│   ├── run_search_benchmark.py
│   ├── run_search.py
│   ├── run_benchmark.py
│   └── compare_recall_methods.py
├── docs/
├── docker-compose.yml
├── .env.example
├── README.md
└── README-CN.md
```

## Reproduction Notes

- Results depend on the actual LLM, embedding, and rerank services configured in `.env`; changing models, embedding dimensions, or rerank settings can change the metrics.
- Storage behavior depends on `STORAGE_PROFILE`. Keep the same profile for initialization, upload, and search unless you intentionally migrate data.
- OceanBase vector search uses `COSINE` distance and returns an ES-compatible `_score` computed from the returned cosine distance. Approximate ANN search uses OceanBase `APPROX LIMIT ... PARAMETERS (ef_search=...)`.
- When `--source-config-id` is omitted, `run_search_benchmark.py` looks up the latest uploaded source based on `LLM_MODEL` in `.env`.
- Full dataset upload and benchmark runs call external model services. Check quota, concurrency, and timeout settings before running.
- Stop local services with `docker compose down`. To delete local database volumes, use `docker compose down -v`; this removes uploaded data.
