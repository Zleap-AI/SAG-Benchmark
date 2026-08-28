"""Adapter registry — central registration of all PredictionAdapter implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.evaluation.judge.errors import AdapterNotFoundError

if TYPE_CHECKING:
    from pipeline.evaluation.judge.adapters.base import PredictionAdapter


class AdapterRegistry:
    """Immutable mapping of project name → PredictionAdapter."""

    def __init__(self) -> None:
        self._adapters: dict[str, PredictionAdapter] = {}

    def register(self, adapter: PredictionAdapter) -> None:
        name = adapter.name
        if not name:
            raise ValueError("Adapter name must not be empty")
        if name in self._adapters:
            raise ValueError(f"Duplicate adapter name: {name}")
        self._adapters[name] = adapter

    def get(self, name: str) -> PredictionAdapter:
        try:
            return self._adapters[name]
        except KeyError:
            raise AdapterNotFoundError(
                f"No adapter registered for project '{name}'. "
                f"Available: {', '.join(sorted(self._adapters))}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def build_default_registry() -> AdapterRegistry:
    """Build the default AdapterRegistry with all supported project adapters."""
    from pipeline.evaluation.judge.adapters.graphrag import GraphRAGAdapter
    from pipeline.evaluation.judge.adapters.hipporag2 import HippoRAG2Adapter
    from pipeline.evaluation.judge.adapters.hypergraphrag import HyperGraphRAGAdapter
    from pipeline.evaluation.judge.adapters.hyperrag import HyperRAGAdapter
    from pipeline.evaluation.judge.adapters.lightrag import LightRAGAdapter
    from pipeline.evaluation.judge.adapters.sag import SAGAdapter

    registry = AdapterRegistry()
    registry.register(GraphRAGAdapter())
    registry.register(HippoRAG2Adapter())
    registry.register(HyperGraphRAGAdapter())
    registry.register(HyperRAGAdapter())
    registry.register(LightRAGAdapter())
    registry.register(SAGAdapter())
    return registry
