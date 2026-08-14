# Search module

This project uses SAG2Searcher as its primary graph-retrieval strategy.
The SAG2 implementation is split into three algorithm stages, an orchestrator,
and a dedicated runtime dependency layer. It does not import another strategy.

## Directory structure

~~~
search/
|-- searcher.py          # Unified strategy entry point
|-- sag2/
|   |-- orchestrator.py  # Request lifecycle, stage composition, result assembly
|   |-- recall.py        # Query rewrite, scope construction, two-route recall
|   |-- expand.py        # Event/entity graph expansion
|   |-- rerank.py        # LLM rank, rerank model, RRF, and fallbacks
|   |-- runtime.py       # LLM, embedding, prompt, storage, and client adapters
|   |-- candidate_scope.py # Optional request-scoped event/entity universe
|   |-- contracts.py     # Typed request, state, and stage result contracts
|   |-- routes.py        # Route edge recording
|   |-- timing.py        # Request-local timing and event statistics
|   |-- evidence.py      # Evidence coverage diagnostics
|   `-- utils.py         # Stateless SAG2 helpers
|-- config.py            # Shared search configuration and SAG2 sub-configs
|-- atomic.py            # ATOMIC compatibility strategy
|-- multi_vector.py      # MULTI_ES compatibility strategy, isolated from SAG2
|-- vector.py            # VECTOR strategy
`-- bm25.py              # BM25 strategy
~~~

## SAG2 dependency boundary

SAG2Runtime owns lazy initialization of the LLM client, document processor,
PromptProvider, storage ports, rerank client, and token counter. Recall,
Expand, and Rerank stages own their respective algorithms. SAG2Searcher only
composes the stages, performs final route/chunk hydration, and assembles the
response diagnostics.

When `sag2_scope.enabled` is true, `sag2/candidate_scope.py` builds the bounded
event/entity candidate universe. The same runtime, expansion, reranking, and
final chunk-hydration boundaries are used in both scoped and global modes.

The compatibility MULTI_ES implementation remains an independent route. It
is not imported by SAG2Searcher and is not used as a base class.

## Active strategies

| Strategy | Implementation |
|---|---|
| SAG2 | `sag2/orchestrator.py` + `sag2/{recall,expand,rerank}.py` |
| MULTI_ES | multi_vector.py retained compatibility route (not actively maintained) |
| ATOMIC | atomic.py |
| VECTOR | vector.py |
| BM25 | bm25.py |

Use `strategy=sag2` for the current primary graph search path.
