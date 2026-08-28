# Upstream provenance

- Upstream: https://github.com/OSU-NLP-Group/HippoRAG
- Integration form: vendored HippoRAG 2 source under `src/hipporag/`
- Audited source commit: `7366a75daf5772aa68ec27ada532478fd1ba7615`
- Upstream license: MIT

Local patches cover OpenAI-compatible endpoints, embedding/index persistence, cost
tracking, benchmark prompts, canonical `outputs/` paths, and reproduce/scripts glue.
The integrated tree contains local changes and is not a byte-for-byte upstream checkout.
Keep upstream refreshes separate from benchmark-specific patches.
