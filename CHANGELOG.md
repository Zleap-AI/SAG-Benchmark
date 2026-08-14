# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [v2.0.0] — 2026-08-14 — paper v2 release candidate

This section describes the major v2 update prepared for the SAG paper experiments. The release tag and GitHub Release should be created after maintainer review and merge.

### Added

- Public benchmark entry points for `sag2`, `atomic`, `vector`, and `bm25`.
- The SAG2 event-candidate-pool variant through `--sag2-scope`.
- A staged SAG2 pipeline covering recall, expansion, reranking, orchestration, timing, and evidence assembly.
- Evaluation and LLM-as-a-Judge workflows for retrieval and QA experiments.
- Compact extraction prompts and related upload/search benchmark workflows.
- OceanBase storage profiles, initialization support, and vector-index configuration.
- Release, development, contribution, security, and support documentation.

### Changed

- Refactored storage interfaces and Elasticsearch backends behind a stable storage boundary.
- Isolated the compatibility `MULTI_ES` route from the SAG2 implementation.
- Updated search, evaluation, benchmark, and configuration documentation for the v2 workflow.
- Replaced the internal MLflow experiment default with the public `sag-benchmark` name.
- Updated the public configuration template and benchmark tests for the v2 layout.

### Compatibility notes

- This is a major version update; existing imports, storage configuration, and benchmark commands may require migration.
- `MULTI_ES` remains a compatibility route and is not the primary maintained graph-search strategy.
- Python 3.11 or newer is required.
- This entry is a release candidate until maintainers review the PR and create the `v2.0.0` tag.

## [v1.0.0] — 2026-08-14 — release baseline

This entry consolidates the major changes accumulated since the early
prototype period and defines the first public event-universe benchmark
baseline. The version is a release baseline until the maintainers create the
corresponding Git tag and publish package metadata.

### Added

- A three-stage SAG2 search architecture: recall, expansion, and reranking.
- Explicit SAG2 stage contracts, runtime services, timing, evidence assembly,
  and focused regression coverage.
- A storage facade with selectable `mysql_es`, `oceanbase_es`, and
  `oceanbase_full` profiles.
- Compact extraction as the default upload mode, with explicit Atomic and
  no-extraction/vector-only alternatives.
- Public search benchmark documentation for the `atomic`, `sag2`, `vector`,
  and `bm25` strategies, including their command-line parameters.
- OceanBase initialization and vector-index support for the applicable storage
  profiles.
- QA benchmark and retrieval-result evaluation workflows, including EM/F1
  evaluation guidance.
- Open-source project documentation: architecture, development, contribution,
  support, security, code of conduct, and pull-request guidance.

### Changed

- Search orchestration now routes SAG2 through its stage package while keeping
  storage and AI runtime concerns behind stable interfaces.
- Deprecated SAG2 compatibility shims and stale numbered runtime entry points
  were removed after the staged implementation was validated.
- The public documentation now distinguishes maintained strategies from
  historical compatibility material.
- The default LLM retry setting was reduced to two attempts to avoid nested
  retry amplification and long waits.
- Benchmark documentation now records compact upload behavior, vector-only
  prerequisites, ablation procedures, and output conventions.

### Fixed

- Restored storage initialization and Event Universe database access through
  the active provider boundary.
- Preserved SAG2 route statistics and stage-level result contracts during the
  refactor.
- Added credential checks and public-documentation safeguards against
  committing access tokens, private infrastructure details, or local paths.

### Compatibility notes

- Existing benchmark outputs remain data artifacts and are not part of the
  source distribution.
- Development-only plans and audits are excluded from the source distribution;
  current commands and APIs are documented by the active README and `docs/`
  pages.
- The package metadata still reports `0.1.0`. It should be updated together
  with the release tag when this `v1.0.0` baseline is formally published.

## [Unreleased]

Future changes should be recorded here first and moved into a numbered
version section when a release is prepared.
