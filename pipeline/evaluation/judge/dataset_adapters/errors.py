"""Backward-compatible imports for dataset adapter exceptions."""

from pipeline.evaluation.judge.errors import (
    AmbiguousGroundTruthMatchError,
    DatasetCapabilityError,
    DatasetError,
    DatasetSchemaError,
    GroundTruthMatchError,
    UnsupportedDatasetError,
)

__all__ = [
    "AmbiguousGroundTruthMatchError",
    "DatasetCapabilityError",
    "DatasetError",
    "DatasetSchemaError",
    "GroundTruthMatchError",
    "UnsupportedDatasetError",
]
