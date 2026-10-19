# Upstream provenance

- Upstream: https://github.com/microsoft/graphrag
- Integration form: released dependency plus SAG-Benchmark reproduce scripts
- Pinned engine: `graphrag==2.0.0`
- Vendored commit: not applicable; the GraphRAG package source is not vendored here
- License: upstream MIT; dependency license remains authoritative

Local ownership covers dataset preparation, indexing/retrieval/QA orchestration, artifact
paths, adapters, and benchmark documentation. Do not copy package internals into this
directory; update the pinned dependency and lockfile in a reviewed change.
