# Development guide

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker Compose for integration services
- Access to the LLM, embedding, and optional rerank endpoints used by your experiment

## Create a development environment

```bash
git clone <repository-url>
cd <repository-directory>
uv sync --extra dev
cp .env.example .env
```

Edit `.env` with local values. Never commit `.env`, API keys, passwords, service URLs that are not public, generated datasets, or benchmark outputs.

## Checks before opening a pull request

Run the same checks used by continuous integration:

```bash
uv run python scripts/check_git_remote_credentials.py
uv run ruff check --config pyproject.toml pipeline scripts tests external/*/reproduce external/*/scripts external/*/*_config.py
uv run ruff format --check --config pyproject.toml pipeline scripts tests external/*/reproduce external/*/scripts external/*/*_config.py
uv run python -m compileall -q pipeline scripts
uv run pytest
```

Some integration tests require the configured database, search backend, or model endpoints. If an integration test cannot run locally, state the limitation and the exact command in the pull request rather than weakening the test or committing local service settings.

For an optional live OceanBase vector smoke check, configure `oceanbase_full` in
the local `.env`, then run:

```bash
uv run python scripts/test_oceanbase_knn.py --query "Who is Lionel Messi?" --top-k 5
```

This command calls the configured embedding service and queries all four vector
tables through `StorageFacade`; it is not part of the offline test suite.

## Safe change workflow

1. Create a focused branch from the current default branch.
2. Keep each commit limited to one logical change.
3. Add or update tests for behavior changes.
4. Update the relevant public documentation and command examples.
5. Review `git diff --check` and scan the diff for secrets and machine-specific paths.
6. Open a pull request with the motivation, scope, validation commands, and any benchmark configuration needed to reproduce the result.

## Project conventions

- Prefer configuration from typed settings/config objects over literals in callers.
- Keep search strategies independent of concrete database clients; use the storage facade/provider boundary.
- Preserve the public CLI and import compatibility unless a breaking change is explicitly documented.
- Keep generated artifacts in ignored output directories.
- Do not place private server addresses, usernames, absolute local paths, or access tokens in source, tests, documentation, or examples.
