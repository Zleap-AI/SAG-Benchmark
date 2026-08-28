# Third-Party Notices

SAG Benchmark includes vendored or adapted research implementations under
`external/`. The source projects remain distinct from the first-party
benchmark pipeline. Their license texts are preserved in `licenses/`.

The repository's MIT license (`LICENSE`) applies to the first-party content of this
repository (source code, scripts, prompts, docs, and assets), with two exclusions:
it does **not** cover the vendored third-party code under `external/` (see the
ledger below) or the redistributed benchmark datasets under `dataset/` (see
[`dataset/README.md`](dataset/README.md)).

## Provenance ledger

The commit column records the source checkout revision used for the current
integration audit. It is a provenance record, not a claim that the external
copy is a clean checkout: each entry may contain local adapters, configuration,
or reproduction scripts.

| Integrated project | Public source URL | Audited source commit | License text | Integration status |
| --- | --- | --- | --- | --- |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag | `1af2dd0d6d777134a4620cba14885e8c7512229a` | [`licenses/GraphRAG-LICENSE`](licenses/GraphRAG-LICENSE) | Vendored engine with local `reproduce/` integration |
| HippoRAG 2 | https://github.com/OSU-NLP-Group/HippoRAG | `7366a75daf5772aa68ec27ada532478fd1ba7615` | [`licenses/HippoRAG-LICENSE`](licenses/HippoRAG-LICENSE) | Adapted copy with local reproduction/configuration code |
| HyperGraphRAG | https://github.com/LHRLAB/HyperGraphRAG | `1cbfc65fd7ba11cfb395eb29c640806e2699a4dd` | [`licenses/HyperGraphRAG-LICENSE`](licenses/HyperGraphRAG-LICENSE) | Adapted copy with local reproduction/configuration code |
| Hyper-RAG | https://github.com/iMoonLab/Hyper-RAG | `5b16b64860a7e6d60f18cda8a2c4d3aee911752a` | [`licenses/Hyper-RAG-LICENSE`](licenses/Hyper-RAG-LICENSE) | Adapted copy with local reproduction/configuration code |
| LightRAG | https://github.com/HKUDS/LightRAG | `2234a6958f8ff2802e60c694e5d27695270260c6` | [`licenses/LightRAG-LICENSE`](licenses/LightRAG-LICENSE) | Adapted copy with local reproduction/configuration code |
| GraphRAG-Benchmark Judge | https://github.com/AI-Data-Robotics-Lab/GraphRAG-Benchmark | `f09ada35eab3e5554a9febc07ccbc41a2b9b4026` | [`licenses/GraphRAG-Benchmark-LICENSE`](licenses/GraphRAG-Benchmark-LICENSE) | Legacy Judge source retained for conversion compatibility |

The integrated sources are committed trees rather than Git submodules. Local
modifications should be compared with the audited source commit before a
release. The GraphRAG-Benchmark Judge row records the declared public source;
maintainers must verify that repository/commit relationship before claiming an
exact upstream match.

## First-party integration boundaries

The first-party code is under `pipeline/`, `scripts/`, and `tests/`. The
`external/*/reproduce/` scripts and method configuration are maintained as
owned integration code; implementation code elsewhere under `external/` should
be treated as third-party code when reviewing changes.
