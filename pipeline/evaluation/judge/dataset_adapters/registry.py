"""Registry for explicit raw dataset adapters."""

from __future__ import annotations

from collections.abc import Iterable

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.errors import UnsupportedDatasetError


class DatasetAdapterRegistry:
    """Resolve canonical dataset names and explicitly declared aliases."""

    def __init__(self, adapters: Iterable[DatasetAdapter] = ()) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        self._aliases: dict[str, str] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DatasetAdapter) -> None:
        descriptor = adapter.descriptor
        name = descriptor.name
        if name in self._adapters or name in self._aliases:
            raise ValueError(f"Duplicate dataset adapter name: {name}")
        for alias in descriptor.aliases:
            if alias in self._adapters or alias in self._aliases or alias == name:
                raise ValueError(f"Duplicate dataset adapter alias: {alias}")
        self._adapters[name] = adapter
        for alias in descriptor.aliases:
            self._aliases[alias] = name

    def resolve_name(self, name: str) -> str:
        """Return the canonical name for a registered name or alias."""
        if name in self._adapters:
            return name
        if name in self._aliases:
            return self._aliases[name]
        available = ", ".join(self.names())
        raise UnsupportedDatasetError(
            f"Unsupported dataset {name!r}. Available datasets: {available}"
        )

    def get(self, name: str) -> DatasetAdapter:
        """Return the adapter registered for a name or alias."""
        return self._adapters[self.resolve_name(name)]

    def names(self) -> tuple[str, ...]:
        """Return canonical names in deterministic order."""
        return tuple(sorted(self._adapters))
