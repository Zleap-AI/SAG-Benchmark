# External Verification Projects

This directory contains 5 independent external projects integrated as verification
components for sag-benchmark. Each source subdirectory is an independent
Python project with its own `pyproject.toml`, `uv.lock`, and `.venv`.

## Projects

| Directory | Source | Role |
|-----------|--------|------|
| `graphrag/` | Microsoft GraphRAG | QA method: GraphRAG global/local search |
| `hipporag2/` | OSU-NLP-Group HippoRAG | QA method: HippoRAG 2 (igraph + Parquet vector retrieval) |
| `hypergraphrag/` | HKUDS HyperGraphRAG | QA method: NetworkX graph + nano-vectordb |
| `hyperrag/` | iMoonLab hyperrag-benchmark | QA method: Hypergraph-DB supergraph + nano-vectordb |
| `lightrag/` | LightRAG | QA method: NetworkX graph + nano-vectordb |

See [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) for tested Python/OS/service boundaries. Each method's `UPSTREAM.md` records source provenance and local patch ownership.

## Environment Model

Each method runs in its own isolated uv environment and reads configuration
from the repository root `.env`:

```bash
cd <repo-root>/external/<method>
uv sync --frozen
uv run --frozen --env-file ../../.env python reproduce/Step_*.py ...
```

Each of the five methods (GraphRAG, HippoRAG2, HyperGraphRAG, Hyper-RAG,
LightRAG) maintains its own `pyproject.toml`, `uv.lock`, and `.venv`.

## 3-Step Flow (HippoRAG2, HyperGraphRAG, Hyper-RAG, LightRAG)

```
Step_0_load_dataset.py -> Step_1_build_index.py -> Step_3_response.py
```

## GraphRAG 4-Step Flow

```
Step_0_prepare_dataset.py -> Step_1_build_index.py -> Step_2_retrieve.py -> Step_3_qa_eval.py
```

## Convert & Judge (from repository root)

Use the native Step 3 root shown below:

| Method | Step 3 input root | Canonical artifact root |
|--------|-------------------|-------------------------|
| GraphRAG | `external/graphrag/outputs` | `external/graphrag/outputs/<ds>/<batch>` |
| HippoRAG2 | `external/hipporag2/outputs/<source-id>/<llm>_<emb>` | `<input-root>/qa_result` (index artifacts live in `external/hipporag2/caches/<source-id>/`) |
| HyperGraphRAG | `external/hypergraphrag/outputs` | `external/hypergraphrag/outputs/<ds>` |
| Hyper-RAG | `external/hyperrag/outputs` | `external/hyperrag/outputs/<ds>` |
| LightRAG | `external/lightrag/outputs` | `external/lightrag/outputs/<ds>` |

```bash
cd <repo-root>

# Convert the newest native response to predictions
uv run --frozen --env-file .env python scripts/run_llm_judge.py convert \
  --project <method> \
  --datasets <ds>

# Run generation and retrieval evaluation from the canonical predictions.
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project <method> \
  --dataset <ds>
```

Judge artifacts are unified under the repository root in a three-layer mirror of
each external method's source-run hierarchy:

```
evaluation/<project>/<dataset>/<source_run_id>/
├── predictions/
│   ├── predictions_<dataset>.json
│   └── conversion_manifest.json
└── llmjudge/<judge_model>/<judge_run_id>/
    ├── generation_results.json
    ├── retrieval_results.json
    ├── indexing_results.json
    ├── summary.json
    └── run_manifest.json
```

`<project>` is one of `graphrag|hipporag2|hypergraphrag|hyperrag|lightrag`;
`<source_run_id>` is a human-readable anchor mirroring the native run directory
(e.g. the `<llm>_<emb>` model dir, the `response/<run-id>` timestamp, or the
`<run-id>-<mode>` pair). The authoritative lineage — the absolute
`source_run_root` and the SHA-256 of every `source_file` — is recorded inside
`conversion_manifest.json`.

New runs never overwrite an existing run directory. To extend an existing run,
use `--resume-run-id <id> --metrics <metric,...>` on the `evaluate` subcommand;
this fills only missing or NaN values.
Add `--force-metrics` to replace only those named metrics. The separate
`--force` flag replaces an entire evaluation kind. Metrics are routed to the
generation or retrieval phase by name.

For Hyper-RAG, add `--mode naive|hyper|hyper-lite` to `convert`. For a named
or non-latest response run, add `--source-run-id <run-id>`. Each project README
contains a copy-paste command with its exact paths.

For smoke tests, add `--top-k K` to `evaluate` or `all` to evaluate only the
first K prediction rows in file order. This is different from
`--context-top-k`, which limits plain-text context chunks per question.

## Key Principles

1. **Independent environments** — each method has its own `.venv` and lockfile.
2. **Repository root `.env`** — all methods read `<repo-root>/.env`.
3. **Data exchange via JSON files** — QA outputs are JSON files consumed by the Judge pipeline.
4. **Isolated method integrations** — method-specific code lives under `external/`.
5. **Sensitive values from environment** — API keys/URLs are read from env vars only, never hardcoded.

## Output & Cache

- Method-local output, cache, and downloaded-data directories are not tracked. The canonical public benchmark fixtures are tracked under the repository-root `dataset/`.
- API keys are provided **only** via environment variables or a local `.env` file — never hardcoded.
