"""Contract tests for the explicit DatasetAdapter registry."""

from dataclasses import dataclass

import pytest

from pipeline.evaluation.judge.dataset_adapters.errors import UnsupportedDatasetError
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetDescriptor,
)
from pipeline.evaluation.judge.dataset_adapters.registry import DatasetAdapterRegistry


@dataclass
class _FakeAdapter:
    descriptor: DatasetDescriptor

    def parse_sample(self, raw: dict, row_index: int) -> CanonicalGroundTruthSample:
        return CanonicalGroundTruthSample(
            dataset=self.descriptor.name,
            id=row_index,
            question=raw["question"],
        )

    def validate_dataset(self, samples: tuple[CanonicalGroundTruthSample, ...]) -> None:
        return None


def test_registry_resolves_canonical_name_and_alias():
    adapter = _FakeAdapter(
        DatasetDescriptor(name="fake", aliases=("fixture_fake",)),
    )
    registry = DatasetAdapterRegistry([adapter])

    assert registry.resolve_name("fake") == "fake"
    assert registry.resolve_name("fixture_fake") == "fake"
    assert registry.get("fixture_fake") is adapter
    assert registry.names() == ("fake",)


def test_registry_rejects_unknown_dataset():
    registry = DatasetAdapterRegistry()

    with pytest.raises(UnsupportedDatasetError, match="unknown"):
        registry.get("unknown")


def test_registry_rejects_duplicate_aliases():
    first = _FakeAdapter(DatasetDescriptor(name="first", aliases=("shared",)))
    second = _FakeAdapter(DatasetDescriptor(name="second", aliases=("shared",)))
    registry = DatasetAdapterRegistry([first])

    with pytest.raises(ValueError, match="shared"):
        registry.register(second)
