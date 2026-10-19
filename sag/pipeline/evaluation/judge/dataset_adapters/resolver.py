"""Resolve dataset file paths and explicit dataset adapter names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.registry import DatasetAdapterRegistry
from pipeline.evaluation.judge.errors import (
    DatasetFileNotFoundError,
    UnsupportedDatasetError,
)


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    requested_name: str
    canonical_name: str
    path: Path
    adapter: DatasetAdapter


class DatasetResolver:
    def __init__(self, registry: DatasetAdapterRegistry) -> None:
        self._registry = registry

    def resolve(self, dataset: str, dataset_dir: Path) -> ResolvedDataset:
        if (
            not isinstance(dataset, str)
            or not dataset.strip()
            or dataset in {".", ".."}
            or Path(dataset).name != dataset
        ):
            raise DatasetFileNotFoundError(f"Invalid dataset name: {dataset!r}")
        dataset_dir = dataset_dir.resolve()
        exact = dataset_dir / f"{dataset}.json"
        path = exact
        candidate = dataset

        if exact.exists():
            adapter = None
            try:
                adapter = self._registry.get(candidate)
            except UnsupportedDatasetError as direct_error:
                for pattern in (r"_\d{8}_\d{6}$",):
                    stripped = re.sub(pattern, "", dataset)
                    if stripped == dataset:
                        continue
                    try:
                        adapter = self._registry.get(stripped)
                    except UnsupportedDatasetError:
                        continue
                    candidate = stripped
                    break
                if adapter is None:
                    raise direct_error
        else:
            adapter = None
            for pattern in (r"_\d{8}_\d{6}$",):
                stripped = re.sub(pattern, "", dataset)
                if stripped == dataset:
                    continue
                candidate_path = dataset_dir / f"{stripped}.json"
                if not candidate_path.exists():
                    continue
                try:
                    adapter = self._registry.get(stripped)
                except UnsupportedDatasetError:
                    continue
                candidate = stripped
                path = candidate_path
                break
            if adapter is None:
                raise DatasetFileNotFoundError(f"Dataset file not found: {exact}")

        return ResolvedDataset(
            requested_name=dataset,
            canonical_name=adapter.descriptor.name,
            path=path.resolve(),
            adapter=adapter,
        )
