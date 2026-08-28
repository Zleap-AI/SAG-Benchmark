# Microsoft GraphRAG — External Verification

Uses the upstream GraphRAG engine (`graphrag==2.0.0`), wrapped in standard
reproduce/ step scripts, feeding results into the shared Judge pipeline.

## Setup

```bash
cd <repo-root>/external/graphrag
uv sync --frozen
```

Configuration is read from the repository root `.env` via
`graphrag_benchmark/config.py`, whose `apply_env()` backfills `os.environ` so the
`${VAR}` placeholders in `settings.yaml` expand when the engine loads its config:

```bash
# <repo-root>/.env
LLM_API_KEY=your-api-key
LLM_MODEL=Qwen3.6-35B-A3B-FP8
LLM_BASE_URL=http://your-llm/v1
EMBEDDING_API_KEY=your-emb-key
EMBEDDING_MODEL_NAME=text-embedding-bge-large-en-v1.5
EMBEDDING_BASE_URL=http://your-emb/v1
```

LLM and embedding run on separate endpoints with separate keys —
`default_chat_model` and `default_embedding_model` in `reproduce/settings.yaml`
each hold their own `api_key` / `api_base` / `model`.

Optional GraphRAG-specific knobs (all have defaults, see `graphrag_benchmark/config.py`):
`GRAPHRAG_ENCODING_MODEL`, `GRAPHRAG_CONCURRENT_REQUESTS`, `GRAPHRAG_MAX_RETRIES`,
`GRAPHRAG_CHUNK_SIZE`, `GRAPHRAG_CHUNK_OVERLAP`, `GRAPHRAG_EMBED_MAX_TOKENS`,
`GRAPHRAG_DISABLE_THINKING`, `GRAPHRAG_MODEL_SUPPORTS_JSON`.

All steps use `uv run --frozen --env-file ../../.env`.

## 4-Step Pipeline

### Step 0 — Prepare Dataset

```bash
uv run --frozen --env-file ../../.env python reproduce/Step_0_prepare_dataset.py --dataset hotpotqa
```

Converts the raw dataset into a CSV corpus + question JSONL, and copies
`reproduce/settings.yaml` into the engine workspace `caches/<ds>/`.
Use `--smoke` for a 50-doc / 20-question dry run.

### Step 1 — Build Index

```bash
uv run --frozen --env-file ../../.env python reproduce/Step_1_build_index.py --dataset hotpotqa
```

Calls `graphrag.api.build_index` in-process to build entities, communities and
embeddings. Prints per-workflow progress with elapsed time and ETA:

```
[3/8 extract_graph]  22/50 (44.0%)  已用 01m00s  预计剩余 01m16s
[3/8 extract_graph]  50/50 (100%)  用时 02m12s  ✓
```

Before indexing, the upstream `validate_config_names` self-check makes one real
call against each endpoint, so a misconfigured endpoint fails fast instead of
retrying silently.

Resumable: an existing `cache/extract_graph/` is kept and the engine skips
already-processed chunks. Re-running with complete `output/` artifacts is a no-op;
pass `--force` to rebuild from scratch. Other flags: `--skip-validation`,
`--progress-interval N` (progress print throttle, default 20s).

### Step 2 — Retrieve

```bash
uv run --frozen --env-file ../../.env python reproduce/Step_2_retrieve.py --dataset hotpotqa
```

Runs GraphRAG local search for each question, emitting the full-graph-context for each question. Prints retrieval progress every 20 questions.

### Step 3 — QA Evaluation

```bash
uv run --frozen --env-file ../../.env python reproduce/Step_3_qa_eval.py --dataset hotpotqa
```

Runs the repository's `scripts/run_qa_benchmark.py` (same QA prompt, answer
extraction and EM/F1 as every other method) over the retrieved full-graph context, then
prints an EM/F1 summary. `--bench-size N` controls the `QA 进度: i/total` cadence.

Note: Step 2 and Step 3 are not resumable — an interrupted run restarts from the
beginning. Only Step 1 supports resume.

## Output Locations

Middle artifacts live in `caches/`, final results in `outputs/` — the same split
the other external methods use.

| Artifact | Path |
|----------|------|
| Engine workspace + config | `caches/<ds>/settings.yaml`, `caches/<ds>/input/<ds>.csv` |
| Prepared data | `caches/<ds>/contexts/`, `caches/<ds>/questions/` |
| GraphRAG index | `caches/<ds>/output/` (parquet + `lancedb/`) |
| Index resume cache | `caches/<ds>/cache/` |
| Engine logs (upstream) | `caches/<ds>/logs/indexing-engine.log`, `logs.json` |
| Cost | `caches/<ds>/cost.json` (index/query phases) |
| Retrieval results | `outputs/<ds>/response/graphrag_<ds>_result.json` |
| QA evaluation | `outputs/<ds>/qa/qa_results.json` |

## Convert & Judge

After Step 3, run the shared Judge from the repository root.

```bash
cd <repo-root>

# 1. Convert Step 3 QA + retrieval output to canonical predictions.
uv run --frozen --env-file .env python scripts/run_llm_judge.py convert \
  --project graphrag \
  --input-root external/graphrag/outputs \
  --datasets <ds> \
  --dataset-dir dataset

# 2. Run generation and retrieval Judge metrics.
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project graphrag \
  --dataset <ds>
```

Judge results are written to the unified repository-root mirror
`evaluation/graphrag/<ds>/<source_run_id>/llmjudge/<judge-model>/<judge-run-id>/`.

## Local Patches

`src/` mirrors files that override the installed `graphrag` package; `_common.py`'s
`ensure_overrides()` syncs them into `.venv` on every step (idempotent, survives
`uv sync`).

| File | Purpose |
|------|---------|
| `src/cost_meter.py` | Token/call accounting per phase → `cost.json` |
| `src/language_model/providers/fnllm/models.py` | Cost hooks + embedding truncation |
| `src/language_model/providers/fnllm/utils.py` | `enable_thinking: false` via `chat_template_kwargs` |
