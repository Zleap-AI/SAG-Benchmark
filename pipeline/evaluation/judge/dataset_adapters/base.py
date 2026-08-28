"""Protocol for explicit raw benchmark dataset adapters."""

from __future__ import annotations

from typing import Any, Protocol

from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetDescriptor,
)


class DatasetAdapter(Protocol):
    """Convert one declared raw dataset schema into canonical samples.

    Implementations must not guess a schema from arbitrary field presence.
    They own a declared dataset name, aliases, schema validation, and the
    conversion rules for that dataset.
    """

    descriptor: DatasetDescriptor

    def parse_sample(
        self,
        raw: dict[str, Any],
        row_index: int,
    ) -> CanonicalGroundTruthSample:
        """Validate and convert one raw row."""
        ...

    def validate_dataset(
        self,
        samples: tuple[CanonicalGroundTruthSample, ...],
    ) -> None:
        """Validate dataset-level invariants after row conversion."""
        ...
