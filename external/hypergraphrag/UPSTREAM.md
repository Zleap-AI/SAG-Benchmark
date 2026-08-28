# Upstream provenance

- Upstream: https://github.com/HKUDS/HyperGraphRAG
- Integration form: vendored package under `hypergraphrag/`
- Local package version: `1.0.6`
- Audited source commit: `1cbfc65fd7ba11cfb395eb29c640806e2699a4dd`
- License: MIT; see `LICENSE`

Local patches cover remote OpenAI-compatible endpoints, optional stores, benchmark
configuration, output layout, graph metrics, and reproduce/scripts glue. Keep upstream
refreshes separate from benchmark-specific patches so the local delta can be reviewed.

## Local patches to upstream source (behavioral)

- `hypergraphrag/utils.py` `limit_async_func_call`: added `try/finally` around the
  wrapped call so the concurrency slot counter (`__current_size`) is released even
  when the wrapped func raises. Upstream bug: an exception (e.g. `APITimeoutError`)
  skipped `__current_size -= 1`, permanently leaking one slot per failure and
  degrading effective LLM concurrency toward 1 over a long index run. Keep this
  patch on any upstream refresh.
