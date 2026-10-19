# External support matrix

This matrix distinguishes repository metadata and CI coverage from full experimental
validation. All commands run from each method's directory with its own `uv.lock`.

| Method | Python declared | CI target | Supported host | Runtime services |
|---|---|---|---|---|
| GraphRAG | 3.10–3.12 | Ubuntu / 3.11, locked install + compile | Linux; Windows not verified | OpenAI-compatible chat and embedding endpoints |
| HippoRAG2 | 3.10+ | Ubuntu / 3.11, locked install + compile | Linux; Windows not verified | OpenAI-compatible chat and embedding endpoints; local graph/vector artifacts |
| HyperGraphRAG | 3.10–3.12 | Ubuntu / 3.11, locked install + compile | Linux; Windows not verified | OpenAI-compatible endpoints; optional storage backends are not CI-covered |
| Hyper-RAG | 3.11+ | Ubuntu / 3.11, locked install + compile | Linux; Windows not verified | OpenAI-compatible endpoints and Hypergraph-DB |
| LightRAG | 3.10–3.12 | Ubuntu / 3.11, locked install + compile | Linux; Windows not verified | OpenAI-compatible chat and embedding endpoints |

## Verification levels

- **CI install/compile:** `uv sync --frozen` plus compilation of the method package,
  `reproduce/`, and `scripts/`.
- **Repository tests:** root offline pytest covers shared adapters and Judge behavior.
- **Experimental smoke:** requires configured services and data; record dataset, source/run
  IDs, model configuration, and artifact paths in the PR.
- **Optional backends:** Neo4j, Milvus, MongoDB, TiDB, Oracle, Chroma, local HF models,
  and similar extras are not supported unless a PR supplies an isolated test.

The root `.env` is local configuration only. Never put real endpoints or credentials in
issues, logs, fixtures, manifests, or committed documentation.
