# Upstream provenance

- Upstream: https://github.com/iMoonLab/Hyper-RAG
- Integration form: vendored package under `hyperrag/`
- Local integration version: `0.1.0`
- Audited source commit: `5b16b64860a7e6d60f18cda8a2c4d3aee911752a`
- License: Apache-2.0; see `LICENSE`

Local patches cover remote endpoints, benchmark configuration, output layout, graph
metrics, and reproduce/scripts glue. Treat `.hgdb` files as trusted local artifacts
only; never use a downloaded database as a test fixture.
