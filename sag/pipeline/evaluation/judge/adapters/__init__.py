"""Judge adapters subpackage — exports protocol, registry, and all adapters."""

from pipeline.evaluation.judge.adapters.base import (
    PredictionAdapter,
    _has_data_files,
    _is_nonempty,
    convert_one,
    ensure_unique_questions,
    find_latest_run,
    normalize_question,
    resolve_dataset_name,
)
from pipeline.evaluation.judge.adapters.registry import (
    AdapterRegistry,
    build_default_registry,
)

__all__ = [
    "AdapterRegistry",
    "PredictionAdapter",
    "build_default_registry",
    "convert_one",
    "find_latest_run",
    "_has_data_files",
    "_is_nonempty",
    "ensure_unique_questions",
    "normalize_question",
    "resolve_dataset_name",
]
