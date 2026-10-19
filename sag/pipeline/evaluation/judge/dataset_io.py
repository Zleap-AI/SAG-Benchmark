"""Dataset I/O backed by explicit adapters and canonical ground-truth models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.evaluation.judge.adapters.base import normalize_question
from pipeline.evaluation.judge.dataset_adapters.defaults import build_default_dataset_registry
from pipeline.evaluation.judge.dataset_adapters.errors import (
    AmbiguousGroundTruthMatchError,
    DatasetCapabilityError,
    DatasetSchemaError,
    GroundTruthMatchError,
)
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetCapability,
    DatasetDescriptor,
)
from pipeline.evaluation.judge.dataset_adapters.registry import DatasetAdapterRegistry
from pipeline.evaluation.judge.dataset_adapters.resolver import (
    DatasetResolver,
    ResolvedDataset,
)


def load_raw_dataset(dataset_name: str, dataset_dir: str) -> list[dict[str, Any]]:
    """Load a registered dataset's raw JSON rows.

    This compatibility helper resolves aliases through the same explicit
    adapter registry used by :class:`GroundTruthRepository`. It does not infer
    a schema from raw field names.
    """
    repository = GroundTruthRepository(Path(dataset_dir))
    return repository.load_raw(dataset_name)


def load_evidence_map(dataset_name: str, dataset_dir: str) -> dict[int, list[str]]:
    """Return canonical gold evidence keyed by the dataset row id."""
    repository = GroundTruthRepository(Path(dataset_dir))
    return {sample.id: list(sample.evidences) for sample in repository.load(dataset_name)}


class GroundTruthRepository:
    """Read-only access to validated canonical ground-truth samples."""

    def __init__(
        self,
        dataset_dir: Path,
        registry: DatasetAdapterRegistry | None = None,
    ) -> None:
        self._dataset_dir = dataset_dir.resolve()
        self._registry = registry or build_default_dataset_registry()
        self._resolver = DatasetResolver(self._registry)
        self._cache: dict[tuple[str, int], tuple[CanonicalGroundTruthSample, ...]] = {}

    def _resolve(self, dataset: str) -> ResolvedDataset:
        return self._resolver.resolve(dataset, self._dataset_dir)

    def resolve_dataset_path(self, dataset: str) -> Path:
        """Resolve a registered dataset alias to its concrete JSON file."""
        return self._resolve(dataset).path

    def load_raw(self, dataset: str) -> list[dict[str, Any]]:
        """Load raw rows after resolving the dataset through its adapter."""
        path = self.resolve_dataset_path(dataset)
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise DatasetSchemaError(f"Dataset {dataset!r} must contain a JSON list of rows")
        if not all(isinstance(row, dict) for row in data):
            raise DatasetSchemaError(f"Dataset {dataset!r} must contain only JSON objects")
        return data

    def _load_canonical(self, dataset: str) -> tuple[CanonicalGroundTruthSample, ...]:
        resolved = self._resolve(dataset)
        stat = resolved.path.stat()
        cache_key = (str(resolved.path), stat.st_mtime_ns)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw_rows = self.load_raw(dataset)
        samples = tuple(
            resolved.adapter.parse_sample(raw, row_index) for row_index, raw in enumerate(raw_rows)
        )
        resolved.adapter.validate_dataset(samples)
        self._cache = {cache_key: samples}
        return samples

    def load(self, dataset: str) -> list[CanonicalGroundTruthSample]:
        """Load validated canonical samples for a registered dataset."""
        return list(self._load_canonical(dataset))

    def descriptor(self, dataset: str) -> DatasetDescriptor:
        """Return the descriptor declared by the resolved dataset adapter."""
        return self._resolve(dataset).adapter.descriptor

    def require_capability(
        self,
        dataset: str,
        capability: DatasetCapability | str,
    ) -> DatasetDescriptor:
        """Fail explicitly when a dataset lacks a requested capability."""
        capability = DatasetCapability(capability)
        descriptor = self.descriptor(dataset)
        if capability not in descriptor.capabilities:
            raise DatasetCapabilityError(
                f"Dataset {descriptor.name!r} does not provide capability {capability.value!r}"
            )
        return descriptor

    def match_dataset_sample_id(
        self,
        dataset: str,
        dataset_sample_id: str,
        question: str,
    ) -> CanonicalGroundTruthSample:
        """Match a prediction by the stable identifier declared by its dataset."""
        matches = [
            sample
            for sample in self._load_canonical(dataset)
            if sample.dataset_sample_id == dataset_sample_id
        ]
        if not matches:
            raise GroundTruthMatchError(
                f"Dataset sample id not found in {dataset!r}: {dataset_sample_id!r}"
            )
        if len(matches) > 1:
            ids = ", ".join(str(sample.id) for sample in matches)
            raise AmbiguousGroundTruthMatchError(
                f"Dataset sample id {dataset_sample_id!r} matches multiple rows in "
                f"dataset {dataset!r}: {ids}"
            )

        match = matches[0]
        if normalize_question(match.question) != normalize_question(question):
            raise GroundTruthMatchError(
                f"Dataset sample id {dataset_sample_id!r} has a different question in "
                f"dataset {dataset!r}"
            )
        return match

    def match_question(
        self,
        dataset: str,
        question: str,
    ) -> CanonicalGroundTruthSample:
        """Match a prediction question to exactly one canonical sample."""
        normalized = normalize_question(question)
        if not normalized:
            raise GroundTruthMatchError("Cannot match an empty question")
        matches = [
            sample
            for sample in self._load_canonical(dataset)
            if normalize_question(sample.question) == normalized
        ]
        if not matches:
            raise GroundTruthMatchError(f"Question not found in dataset {dataset!r}: {question!r}")
        if len(matches) > 1:
            ids = ", ".join(str(sample.id) for sample in matches)
            raise AmbiguousGroundTruthMatchError(
                f"Question matches multiple rows in dataset {dataset!r}: {ids}"
            )
        return matches[0]

    def match_canonical_id(
        self,
        dataset: str,
        sample_id: int,
        question: str,
    ) -> CanonicalGroundTruthSample:
        """Match a canonical prediction row by its dataset row id.

        The prediction id is the canonical dataset row index after conversion.
        It is the only unambiguous identity when a dataset contains duplicate
        question text.
        """
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
            raise GroundTruthMatchError(
                f"Dataset row id must be a non-negative integer, got {sample_id!r}"
            )

        samples = self._load_canonical(dataset)
        if sample_id >= len(samples):
            raise GroundTruthMatchError(f"Dataset row id not found in {dataset!r}: {sample_id}")

        match = samples[sample_id]
        if normalize_question(match.question) != normalize_question(question):
            raise GroundTruthMatchError(
                f"Dataset row id {sample_id} has a different question in dataset {dataset!r}"
            )
        return match

    def match_by_question(self, dataset: str) -> dict[str, dict[str, Any]]:
        """Return a question index, rejecting ambiguity instead of selecting a row."""
        result: dict[str, dict[str, Any]] = {}
        for sample in self._load_canonical(dataset):
            key = normalize_question(sample.question)
            if key in result:
                raise AmbiguousGroundTruthMatchError(
                    f"Question matches multiple rows in dataset {dataset!r}: "
                    f"{result[key]['id']}, {sample.id}"
                )
            result[key] = {
                "id": sample.id,
                "ground_truth": sample.answer,
                "question": sample.question,
                "evidences": list(sample.evidences),
            }
        return result

    def by_id(self, dataset: str) -> dict[int, str | list[str]]:
        """Return canonical answers keyed by row id."""
        return {sample.id: sample.answer for sample in self._load_canonical(dataset)}

    def evidence_map(self, dataset: str) -> dict[int, tuple[str, ...]]:
        """Return canonical evidence keyed by row id."""
        return {sample.id: sample.evidences for sample in self._load_canonical(dataset)}


def load_predictions(path: str) -> list[dict[str, Any]]:
    """Load predictions JSON file (list of samples)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_question_type(
    data: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group prediction items by question_type."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in data:
        q_type = item.get("question_type", "Uncategorized")
        grouped.setdefault(q_type, []).append(item)
    return grouped
