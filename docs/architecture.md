# Architecture overview

SAG Benchmark is a reproducible retrieval and question-answering benchmark. The repository separates the benchmark workflow into loading, extraction, storage, search, and evaluation layers so that a dataset can be indexed once and compared across multiple retrieval strategies.

## End-to-end flow

```text
dataset JSON
    -> loader and chunker
    -> extraction prompt (Compact by default, Atomic optional)
    -> storage facade
         -> SQL metadata and event/entity relations
         -> vector/search backend
    -> search strategy
         -> candidate recall
         -> optional event expansion
         -> ranking and evidence assembly
    -> benchmark metrics
    -> optional QA generation and EM/F1 evaluation
```

## Main boundaries

| Area | Responsibility | Typical location |
| --- | --- | --- |
| Application configuration | Load environment variables and validate typed settings | `pipeline/core/config/` |
| AI clients | LLM, embedding, rerank, retry, and usage tracking adapters | `pipeline/core/ai/` |
| Loading and extraction | Read datasets, create chunks, and extract events/entities | `pipeline/modules/load/`, `pipeline/modules/extract/` |
| Storage facade | Select the configured backend and expose stable application operations | `pipeline/storage/` |
| Search orchestration | Route a strategy and coordinate its lifecycle | `pipeline/modules/search/searcher.py` |
| SAG2 stages | Recall, expansion, reranking, and evidence assembly | `pipeline/modules/search/sag2/` |
| Search strategies | SAG2, atomic, vector, and BM25 behavior | `pipeline/modules/search/` |
| Evaluation | Dataset loading, retrieval metrics, QA metrics, and result persistence | `pipeline/evaluation/` |
| User entry points | Upload, search, benchmark, and database/index initialization | `scripts/` |

## Storage profiles

The application talks to one storage facade. `STORAGE_PROFILE` selects the concrete composition:

| Profile | Structured store | Vector/search store |
| --- | --- | --- |
| `mysql_es` | MySQL | Elasticsearch |
| `oceanbase_es` | OceanBase | Elasticsearch |
| `oceanbase_full` | OceanBase | OceanBase vector search |

Search strategies should depend on the facade/provider interfaces rather than constructing backend clients directly. Backend-specific clients belong under the storage backend implementation, which keeps strategy code portable across profiles.

## SAG2 stages

SAG2 is organized as three algorithm stages with runtime and storage services injected around them:

1. **Recall** identifies entity- and event-based candidates from the query.
2. **Expand** follows event/entity relations to add controlled multi-hop candidates.
3. **Rerank** scores candidates, applies the configured final-selection policy, and assembles evidence chunks.

The stage boundaries make each step testable and allow the search entry point to report stage timings without coupling the algorithm to a particular database or LLM SDK.

## Configuration and reproducibility

Use `.env.example` as the public configuration template and keep secrets only in an untracked `.env` file. For comparable experiments, pin the dataset, source configuration, embedding model and dimension, strategy options, `k-values`, and output directory. Do not commit generated datasets, benchmark outputs, credentials, or service-specific local configuration.
