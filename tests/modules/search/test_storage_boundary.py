"""Architecture guardrails for search-to-storage dependencies."""

from pathlib import Path


def test_search_implementations_do_not_import_legacy_storage_namespace():
    search_dir = Path("pipeline/modules/search")
    violations = []
    for path in sorted(search_dir.glob("*.py")):
        if "pipeline.storage.backends" in path.read_text(encoding="utf-8"):
            violations.append(path.as_posix())

    assert violations == [], (
        "Search implementations must depend on pipeline.storage ports/providers, "
        f"not legacy storage internals: {violations}"
    )


def test_legacy_core_storage_sources_are_removed():
    assert list(Path("pipeline/core/storage").rglob("*.py")) == []


def test_only_storage_infrastructure_imports_backend_internals():
    allowed_files = {
        "pipeline/storage/indexing.py",
        "scripts/init_elasticsearch.py",
    }
    allowed_prefixes = (
        "pipeline/storage/backends/",
        "pipeline/storage/providers/",
    )
    violations = []

    for root in (Path("pipeline"), Path("scripts")):
        for path in sorted(root.rglob("*.py")):
            normalized = path.as_posix()
            if normalized in allowed_files or normalized.startswith(allowed_prefixes):
                continue
            if "pipeline.storage.backends" in path.read_text(encoding="utf-8"):
                violations.append(normalized)

    assert violations == [], (
        "Backend internals may only be used by providers, the public indexing "
        f"adapter, and backend maintenance scripts: {violations}"
    )
