"""Base adapter protocol and shared utilities.

PredictionAdapter Protocol replaces the bare-function registration.
Concrete adapters implement locate_source() and convert().
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Protocol

from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PredictionAdapter(Protocol):
    """Protocol for project-specific native-result adapters.

    Each adapter only reads its own native output format and returns
    a list of rows. It never writes predictions, calls LLMs, or creates
    directories.
    """

    name: str

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        """Find the exact source run directory and list its files."""
        ...

    def convert(self, source: SourceRun) -> AdapterConversion:
        """Read native results and return rows + metadata."""
        ...


# ---------------------------------------------------------------------------
# Shared pure functions (no project-specific behaviour)
# ---------------------------------------------------------------------------


def _is_nonempty(val: Any) -> bool:
    """Return whether a scalar or multi-gold value contains meaningful text."""
    if val is None:
        return False
    if isinstance(val, list):
        return any(v.strip() for v in val if isinstance(v, str))
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)


def normalize_question(q: str) -> str:
    """Normalise question text for comparison."""
    return q.strip()


def ensure_unique_questions(rows: list[dict[str, Any]], adapter_name: str) -> None:
    """Raise if question text is duplicated in rows."""
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        q = normalize_question(row.get("question", ""))
        if q in seen:
            from pipeline.evaluation.judge.errors import AdapterConversionError

            raise AdapterConversionError(
                f"{adapter_name}: duplicate question at rows {seen[q]} and {i}: {q!r}"
            )
        seen[q] = i


def resolve_dataset_name(dataset: str, dataset_dir: Path) -> str:
    """Resolve dataset name to an existing JSON file name (no extension)."""
    ds_path = dataset_dir / f"{dataset}.json"
    if ds_path.exists():
        return dataset
    for pattern in [r"_\d{8}_\d{6}$", r"\d+$"]:
        base = re.sub(pattern, "", dataset)
        if base != dataset:
            alt = dataset_dir / f"{base}.json"
            if alt.exists():
                return base
    return dataset


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


class _LazyAdapterDict(dict):
    """Lazy-loading adapter dict — resolves adapter by project name on access."""

    _BUILT: bool = False
    _REGISTRY: Any = None

    def _ensure_built(self) -> None:
        if self._BUILT:
            return
        from pipeline.evaluation.judge.adapters.registry import build_default_registry

        self._REGISTRY = build_default_registry()
        self._BUILT = True

    def __getitem__(self, key: str) -> Any:
        self._ensure_built()
        return self._REGISTRY.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_built()
        try:
            return self._REGISTRY.get(key)
        except Exception:
            return default

    def __contains__(self, key: object) -> bool:
        self._ensure_built()
        return key in self._REGISTRY.names()

    def __iter__(self):
        self._ensure_built()
        return iter(self._REGISTRY.names())

    def __len__(self) -> int:
        self._ensure_built()
        return len(self._REGISTRY.names())


ADAPTERS: dict[str, Any] = _LazyAdapterDict()


def _has_data_files(dirpath: str) -> bool:
    """Check if directory contains expected data files."""
    if not os.path.isdir(dirpath):
        return False
    if os.path.isdir(os.path.join(dirpath, "response")):
        return True
    if any(f.startswith("results_") and f.endswith(".json") for f in os.listdir(dirpath)):
        return True
    if any(f.startswith("hybrid_") and f.endswith("_result.json") for f in os.listdir(dirpath)):
        return True
    if os.path.isdir(os.path.join(dirpath, "qa_result")):
        return True
    for sub in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, sub, "qa_result")):
            return True
    if os.path.isdir(os.path.join(dirpath, "evaluation")):
        return True
    return False


def find_latest_run(input_root: str, dataset: str) -> str | None:
    """Find the latest run directory under input_root.

    This is a legacy helper kept for backward compatibility.
    New code should use concrete adapter locate_source().
    """
    # Prefix match first (pure data dirs)
    prefix_match = None
    if os.path.isdir(input_root):
        candidates = sorted(
            [
                d
                for d in os.listdir(input_root)
                if os.path.isdir(os.path.join(input_root, d))
                and d.startswith(f"{dataset}_")
                and not d.startswith("LlmJudge_")
            ],
            reverse=True,
        )
        if candidates:
            prefix_match = os.path.join(input_root, candidates[0])

    ds_dir = os.path.join(input_root, dataset)
    if not os.path.isdir(ds_dir):
        if prefix_match and os.path.isdir(prefix_match):
            ds_dir = prefix_match
        if not os.path.isdir(ds_dir):
            return None
    subdirs = sorted(
        (
            os.path.join(ds_dir, d)
            for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d)) and not d.startswith("LlmJudge_")
        ),
        reverse=True,
    )
    for subdir in subdirs:
        if _has_data_files(subdir):
            return subdir
    if _has_data_files(ds_dir):
        return ds_dir
    if prefix_match and os.path.isdir(prefix_match) and _has_data_files(prefix_match):
        print(
            f"    i {dataset}/ exists but no data files, "
            f"falling back to prefix match: {prefix_match}"
        )
        return prefix_match
    return None


def convert_one(
    project: str,
    dataset: str,
    input_root: str,
    dataset_dir: str,
    out_root: str,
    adapters: dict[str, Any] | None = None,
) -> str | None:
    """Legacy convert_one — delegates to PredictionConversionService.

    Kept as a thin compatibility wrapper. New callers should use
    PredictionConversionService directly.
    """
    from pipeline.evaluation.judge.adapters.registry import build_default_registry
    from pipeline.evaluation.judge.artifacts import (
        ArtifactLayoutResolver,
        JudgeArtifactStore,
    )
    from pipeline.evaluation.judge.conversion import PredictionConversionService
    from pipeline.evaluation.judge.dataset_io import GroundTruthRepository

    input_root_p = Path(input_root).resolve()
    dataset_dir_p = Path(dataset_dir).resolve()
    artifact_root = Path(out_root).resolve()

    registry = build_default_registry()
    if adapters:
        from pipeline.evaluation.judge.adapters.registry import AdapterRegistry

        custom = AdapterRegistry()
        # Wrap callables as function adapters
        for name, fn in adapters.items():
            from pipeline.evaluation.judge.models import (
                AdapterConversion,
                ConversionRequest,
                SourceRun,
            )

            class _FnAdapter:
                pass

            _FnAdapter.name = name

            def _locate(self, request, _name=name):
                run_root = Path(request.input_root).resolve()
                return SourceRun(
                    project=_name,
                    dataset=request.dataset,
                    run_root=run_root,
                    artifact_run_root=run_root,
                )

            def _convert(self, source, _fn=fn, _name=name):
                rows = _fn(str(source.run_root), source.dataset)
                return AdapterConversion(rows=tuple(rows))

            _FnAdapter.locate_source = _locate
            _FnAdapter.convert = _convert
            custom.register(_FnAdapter())
        if name in registry.names():
            registry = custom
        else:
            # Merge custom into default: custom takes priority
            for n in custom.names():
                registry._adapters[n] = custom._adapters[n]
    gt_repo = GroundTruthRepository(dataset_dir_p)
    resolver = ArtifactLayoutResolver()
    store = JudgeArtifactStore()

    service = PredictionConversionService(
        registry=registry,
        ground_truth_repository=gt_repo,
        resolver=resolver,
        store=store,
    )

    request = ConversionRequest(
        project=project,
        dataset=dataset,
        input_root=input_root_p,
        dataset_dir=dataset_dir_p,
        artifact_run_root=artifact_root,
        allow_overwrite=True,
    )

    try:
        result = service.convert(request)
        return str(result.predictions_path)
    except Exception as exc:
        print(f"  Conversion failed: {exc}")
        return None
