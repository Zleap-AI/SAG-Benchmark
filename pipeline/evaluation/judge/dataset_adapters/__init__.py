"""Explicit adapters for raw benchmark dataset schemas.

This package is intentionally separate from judge.adapters. The latter
converts native outputs from external RAG projects; this package converts raw
benchmark rows into canonical ground-truth samples.
"""

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetCapability,
    DatasetDescriptor,
)
from pipeline.evaluation.judge.dataset_adapters.registry import DatasetAdapterRegistry

__all__ = [
    "CanonicalGroundTruthSample",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "DatasetCapability",
    "DatasetDescriptor",
]
