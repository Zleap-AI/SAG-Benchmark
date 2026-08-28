<p align="center">
  <img src="assets/logo.svg" alt="Zleap AI" width="220" />
</p>

# SAG Benchmark

> Companion benchmark reproduction repository for the SAG paper. This repository is for reproducing the paper benchmark scores with the quick-start commands. General users, please see the [SAG project](https://github.com/Zleap-AI/SAG).

English | [中文](README-CN.md)

**Paper:** [https://arxiv.org/abs/2606.15971](https://arxiv.org/abs/2606.15971)

https://github.com/user-attachments/assets/ac805e3c-ab52-4857-bef6-2865f3831b2f


## Benchmark Score Reproduction

This repository contains upload, retrieval, and evaluation scripts for SAG on HotpotQA, 2WikiMultiHopQA, and MuSiQue. The current quick-start workflow reproduces the SAG, BM25, and bge-large-en-v1.5 vector retrieval results, and supports the Triple indexing ablation through the atomic upload/search path.

Default paper setup:

| Item | Value |
|------|------|
| Embedding | `bge-large-en-v1.5` |
| LLM | `qwen3.6-flash` |
| Main paper metrics | Recall@5 / F1 |
| Main scripts | `scripts/run_upload.py`, `scripts/run_search_benchmark.py`, `scripts/run_qa_benchmark.py` |

Main results:

<p align="center">
  <img src="assets/main-result.png" alt="SAG main benchmark results" width="760" />
</p>

Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue show that SAG achieves the best retrieval and end-to-end QA performance on every benchmark.

- Across the three datasets, SAG averages **90.07%/72.96%** in Recall@5 and F1, outperforming the strongest baseline for each metric by **6.79/4.33** percentage points, respectively.
- On the most challenging MuSiQue dataset, SAG outperforms the strongest baseline for each metric by **11.52/7.01** percentage points in Recall@5 and F1, respectively.

> **Note:** The current CLI supports reproducing SAG2, BM25, and the bge-large-en-v1.5 vector retrieval results. Atomic indexing/search is available for the Triple indexing ablation.

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
- Available LLM and embedding endpoints; a rerank endpoint is only needed for the final-selection reranker ablation

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

Edit `.env` and fill in the storage backend, LLM, and embedding settings. Standard SAG reproduction only needs these three groups of settings. Add the rerank settings only when running the final-selection reranker ablation described below.

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
```

Only add the following settings for the final-selection reranker ablation:

```env
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

**How `init_elasticsearch.py` resolves the embedding dimension**

By default the script probes the configured embedding model directly to determine the actual vector dimension, then creates four physical indices with a dimension suffix (`source_chunks_4096`, `event_vectors_4096`, etc.). The result is cached in `.cache/embedding_dims.json` so subsequent runs skip the network call.

```bash
# Standard usage — auto-probes the embedding model from .env
uv run python scripts/init_elasticsearch.py

# Force a fresh probe, ignoring the cache
uv run python scripts/init_elasticsearch.py --refresh-dim

# Skip the probe entirely and use a known dimension
uv run python scripts/init_elasticsearch.py --dim 4096
```

The script is idempotent: if an index already exists with a matching dimension it is skipped; if it exists with a mismatched dimension the script fails fast with a clear error rather than silently leaving a broken index. The legacy 1024-dimension indices (`source_chunks`, etc., without a suffix) are preserved untouched when `ES_INDEX_LEGACY_UNSUFFIXED=true` in `.env`.

`run_upload.py` verifies that the required indices exist and have the correct dimension before writing any data. If the indices are missing or were auto-created by Elasticsearch with the wrong mapping, the upload exits immediately with an actionable error message.

### 4. Upload Datasets

Before uploading, prepare the dataset files:

- `test_hotpotqa` and `sample` are included in the repository.
- For `hotpotqa`, `musique`, `2wikimultihopqa`, or `narrativeqa`, download the
  corresponding `*.json` and `*_corpus.json` files from the
  [HippoRAG 2 data release](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
  and place them in `dataset/`. See [dataset/README.md](dataset/README.md) for
  the complete file layout, provenance, licenses, and citation requirements.

`run_upload.py` first converts `dataset/<dataset>.json` into a Markdown corpus, then writes structured rows and vectors through the configured storage facade. With `mysql_es`, data goes to MySQL plus Elasticsearch. With `oceanbase_full`, structured data and vectors are written into OceanBase. After upload, it generates:

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

Upload has three supported modes:

| Mode | Command | Behavior |
|------|---------|----------|
| Compact (default) | `uv run python scripts/run_upload.py --dataset musique` | Uses the compact extraction prompt and creates merged events. No extra flag is required. |
| Atomic | `uv run python scripts/run_upload.py --dataset musique --atomic` | Uses the atomic-event template; each event is constrained to a subject-relation-object pair. |
| No extraction (Vector-only) | `uv run python scripts/run_upload.py --dataset musique --no-extraction` | Loads and indexes the corpus without event/entity extraction; use this when you only want the pure embedding `vector` search path. |

Choose one mode per upload. The default compact mode is the recommended path for SAG2 and for experiments that compare SAG2 with vector or BM25. Use `--no-extraction` only for a vector-only run when SAG is not part of the experiment; SAG2 requires extracted events, so do not use `--no-extraction` if you plan to run SAG2 as well.


### 5. Run Search

#### 5.1 Minimal Command

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy sag2
```

The omitted options use their built-in defaults: `top-k=10`, `k-values=1,2,5,10`, `max-concurrency=10`, and `bench-size=5`. These defaults are the paper configuration, so this single command is the paper's SAG2 retrieval reproduction path. Replace `musique` with `hotpotqa` or `2wikimultihopqa` for the other main datasets.

#### 5.2 Common Parameter Overrides

Append these to the 5.1 command as needed:

```bash
# Adjust concurrency (default 10)
--max-concurrency 20

# Run only the first N questions (unlimited by default)
--limit 10

# Pin a specific uploaded source (by default the latest upload is picked; the ID comes from the source_info.json generated in Step 4)
--source-config-id <source_config_id>
```

See [docs/search.md](docs/search.md) for the full argument reference.

#### 5.3 Other Strategies and Variants

The commands below list only parameters that differ from the defaults; everything else matches 5.1.

##### SAG2 Scope (Event Candidate Pool)

Enable the candidate pool and set its size `k_pool` with `--sag2-event-top-k`. This variant is recorded as a separate search configuration at the same experiment level as the vector and atomic baselines:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy sag2 \
  --sag2-scope \
  --sag2-event-top-k 500
```

##### Atomic (Triple indexing)

Upload in atomic mode first, then search with the atomic strategy:

```bash
uv run python scripts/run_upload.py --dataset musique --atomic
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy atomic
```

##### BM25

Elasticsearch BM25 keyword-search baseline, no extra options:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy bm25
```

##### Vector

Pure vector retrieval runs directly after the default Compact upload:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy vector
```

If the experiment compares Vector with SAG2, upload with the default Compact mode instead so the same source contains the events required by SAG2. When Vector is the sole method being run, you may upload with `--no-extraction` to skip event extraction, but that source can no longer run SAG2 and would need a re-upload; see [docs/search.md](docs/search.md) for the trade-off.

#### 5.4 Tool Integrations

Enable MLflow experiment tracking:

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy sag2 \
  --use-mlflow \
  --mlflow-url http://localhost:5000 \
  --mlflow-experiment sag-benchmark
```

#### Output

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

### 6. Run QA on Retrieval Results

`run_qa_benchmark.py` consumes the `search_results.json` produced by a retrieval run, asks the configured LLM to answer each question using the retrieved passages, and writes EM/F1 results to a new `qa_<timestamp>/` directory next to the input file.

```bash
uv run python scripts/run_qa_benchmark.py \
  --dataset-name musique \
  --input output/musique/sag2/YOUR_SEARCH_RUN_ID/search_results.json \
  --qa-top-k 5
```

`--qa-top-k` controls how many retrieved passages are placed in each QA prompt. The main output is `qa_results.json`; use `--output-dir` to choose a different output directory, or `--max-concurrency` and `--limit` to control runtime and scope.

### 7. Run External Methods

The external baselines are independent integrations. Their setup, indexing, retrieval, QA, output layout, and method-specific options are documented in their own READMEs. Run the commands from the relevant project directory and use the repository-root .env as described there.

| Method | Usage guide |
|---|---|
| Microsoft GraphRAG | [external/graphrag/README.md](external/graphrag/README.md) |
| HippoRAG 2 | [external/hipporag2/README.md](external/hipporag2/README.md) |
| HyperGraphRAG | [external/hypergraphrag/README.md](external/hypergraphrag/README.md) |
| Hyper-RAG | [external/hyperrag/README.md](external/hyperrag/README.md) |
| LightRAG | [external/lightrag/README.md](external/lightrag/README.md) |

See [external/README.md](external/README.md) for the integrated-method overview. After a method has produced its native results, use the Judge workflow below to convert and evaluate them in the unified artifact layout.

### 8. Run LLM Judge

run_llm_judge.py converts a method's latest native output into the shared prediction format and evaluates generation, retrieval, and optionally indexing metrics. At the end of each run it prints every requested metric value and its valid-sample count.

Run the complete convert + generation + retrieval flow:

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py \
  --project hypergraphrag \
  --dataset musique
```

See [docs/judge.md](docs/judge.md) for the full subcommand reference, argument table, metric routing, and output layout.

To run evaluation separately, convert the native output first, then select the metrics needed. The `evaluate` subcommand runs generation and/or retrieval evaluation in a single Judge run; metrics are routed to the correct phase by name:

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py convert \
  --project hypergraphrag \
  --datasets musique

uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset musique \
  --metrics context_relevancy,evidence_recall
```

For core SAG, use `--project sag`. It is fixed to the `sag2` strategy and automatically selects the latest **complete** run in `output/<dataset>/sag2/<run_id>/`: a run must contain both `search_results.json` and `qa_*/qa_results.json`. Run the QA benchmark first, then run:

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py \
  --project sag \
  --dataset musique
```

Generation metrics are answer_correctness, coverage_score, qa_em, qa_f1, and rouge_score; retrieval metrics are context_relevancy and evidence_recall. `--metrics` accepts any mix of these across both phases and routes each metric to its phase automatically; when omitted, all generation and retrieval metrics run by default (evidence_recall is dropped automatically for datasets without evidence capability). Adding the `indexing` token to `--metrics` runs graph indexing metrics in the same Judge run (requires `--framework` and `--base-path`); without it, indexing is skipped by default.

The `indexing` subcommand also exists on its own to backfill graph indexing metrics into an existing Judge run without re-running generation or retrieval. HyperGraphRAG graphs are graphml; the supported `--framework` values are listed in [docs/judge.md](docs/judge.md):

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py indexing \
  --project hypergraphrag \
  --dataset musique \
  --framework graphml \
  --base-path /path/to/graph/output \
  --resume-run-id YOUR_JUDGE_RUN_ID
```

#### Recover failed Judge samples

Each result file records per-sample status, error type, and error message in its `detailed` field. First correct the underlying cause (for example, an API endpoint or model configuration issue), then resume the same Judge run with `--retry-failed`. Successful samples and already valid metric results are retained; only previously failed samples are requested again.

```bash
# Retry failed samples only (retrieval metrics are selected by name).
# Use the Judge run ID printed by the original command.
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset musique \
  --metrics context_relevancy,evidence_recall \
  --resume-run-id YOUR_JUDGE_RUN_ID \
  --retry-failed
```

Use `evaluate` instead of the retired `generation`/`retrieval` subcommands (generation-only, retrieval-only, and indexing selections are all expressed via `--metrics`). `--retry-failed` requires `--resume-run-id` and cannot be combined with `--force` or `--force-metrics`; `--top-k` caps the number of evaluated rows, it is not the retrieval k. Resume is intentionally unavailable in the no-subcommand auto mode. See [docs/judge.md](docs/judge.md) for the full argument constraints.

`--top-k` is a limit on the first N prediction rows evaluated; it does **not** change the retrieval result k. `--context-top-k` controls how many already-saved contexts each Judge metric sees. Both are part of the resume contract: retry with the same values used by the original run. To change either value, create a new Judge run rather than combining the change with `--retry-failed`.

### 9. Reproduce SAG Ablation Results

The following experiments correspond to Table 6 of the paper. Run them on MuSiQue with the same dataset, source configuration, embedding model, and `top-k` values, changing only the indicated component.

#### Default SAG

Use the default Compact upload and the `sag2` search strategy. The default final selection is `llm_rank`.

```bash
uv run python scripts/run_upload.py --dataset musique
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy sag2 \
  --top-k 10 \
  --k-values "1,2,5,10"
```

#### Indexing with Triple indexing

Upload with the atomic prompt and select the `atomic` search strategy:

```bash
uv run python scripts/run_upload.py --dataset musique --atomic
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy atomic \
  --top-k 10 \
  --k-values "1,2,5,10"
```

#### Expansion without expansion (`L=0`)

In `pipeline/modules/search/config.py`, temporarily change the SAG2 expansion switch from:

```python
enabled: bool = Field(default=True, description="是否启用扩展")
```

to:

```python
enabled: bool = Field(default=False, description="是否启用扩展")
```

Then upload with the default Compact mode and run `--strategy sag2`. Restore `default=True` after the ablation so normal SAG2 runs keep one-hop expansion.

#### Final selection with Qwen3-Reranker-8B

In `pipeline/modules/search/config.py`, temporarily change `SAG2RerankConfig.strategy` from the default LLM selection:

```python
strategy: Literal["rerank", "llm_rank", "rrf"] = Field(
    default="llm_rank", description="排序策略"
)
```

to:

```python
strategy: Literal["rerank", "llm_rank", "rrf"] = Field(
    default="rerank", description="排序策略"
)
```

Configure `RERANK_BASE_URL`, `RERANK_MODEL_NAME=Qwen/Qwen3-Reranker-8B`, and `RERANK_ENDPOINT` in `.env`, then run the default Compact upload and `--strategy sag2`. Restore `default="llm_rank"` after the experiment.

#### Paper Ablation Results on MuSiQue

| Stage | Configuration | R@1 | R@2 | R@5 | R@10 |
|------|------|------:|------:|------:|------:|
| Default SAG (Ours) | Default configuration | **36.82** | **63.62** | **80.36** | **83.37** |
| Indexing | Triple indexing | 35.66 | 61.83 | 77.61 | 81.54 |
| Expansion | w/o Expansion (`L=0`) | 35.70 | 57.75 | 69.41 | 74.76 |
| Final selection | Qwen3-Reranker-8B | 32.12 | 48.97 | 67.11 | 76.51 |

The default row uses hyperedge indexing, one-hop expansion (`L=1`), and Qwen3.6-Flash final selection. Keep all non-ablated settings fixed when comparing rows.

## Datasets

| Name | Description |
|------|------|
| `hotpotqa` | HotpotQA multi-hop QA |
| `2wikimultihopqa` | 2WikiMultiHopQA |
| `musique` | MuSiQue multi-hop QA |
| `test_hotpotqa` | Small HotpotQA test set |
| `sample` | Tiny sample set for pipeline debugging |

Dataset files are under `dataset/`. The repository ships only `sample` and
`test_hotpotqa`; download the full benchmarks (`hotpotqa`, `musique`,
`2wikimultihopqa`, `narrativeqa`) from the
[HippoRAG 2 data release](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
and place them in `dataset/` — see [dataset/README.md](dataset/README.md) for
provenance and license notes.

## Retrieval Strategies

| Strategy | Description |
|------|------|
| `sag2` | SAG2 graph recall, expansion, and LLM reranking |
| `sag2` + `--sag2-scope` | SAG2 event-candidate-pool variant; `--sag2-event-top-k` sets `k_pool` |
| `atomic` | Entity-first atomic retrieval with step-by-step hop expansion |
| `vector` | Pure vector retrieval baseline |
| `bm25` | Elasticsearch BM25 keyword retrieval baseline |

See [docs/search.md](docs/search.md) for full arguments.

## Common Scripts

### Search Only

```bash
uv run python scripts/run_search.py \
  --dataset-name test_hotpotqa \
  --strategy sag2 \
  --output-dir output/manual-search
```

### Evaluate Existing Results

```bash
uv run python scripts/run_benchmark.py \
  --results output/<dataset>/<strategy>/<timestamp>/search_results.json \
  --dataset musique
```

## Repository Layout

```text
SAG-Benchmark/
├── assets/                         # README figures and logo
├── dataset/                        # Evaluation datasets
├── pipeline/
│   ├── core/                       # Config, AI clients, storage layer
│   ├── db/                         # SQLAlchemy ORM
│   ├── evaluation/
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
│   ├── run_qa_benchmark.py
│   ├── run_search.py
│   └── run_benchmark.py
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
