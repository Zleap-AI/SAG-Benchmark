"""Protocols for database and vector/search backends."""

from typing import Any, Protocol

from pipeline.storage.capabilities import StorageCapabilities


class DatabaseStore(Protocol):
    """Structured/canonical database backend."""

    backend_name: str

    async def health_check(self) -> bool:
        """Return True when the structured database is reachable."""
        ...

    async def close(self) -> None:
        """Close database resources owned by this store."""
        ...


class EventUniverseStore(Protocol):
    """Canonical relational reads used by the SAG2 event universe.

    This protocol deliberately exposes relation reads separately from
    ``SearchStore``. The event-universe algorithm owns mapping, ordering,
    deduplication, and limits; the provider only returns canonical rows.
    """

    backend_name: str

    async def filter_active_event_ids(
        self,
        event_ids: list[str],
        source_config_ids: list[str] | None = None,
    ) -> list[str]:
        """Return active event ids in the same order as ``event_ids``."""
        ...

    async def get_event_entity_pairs_by_events(
        self,
        event_ids: list[str],
        source_config_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return canonical ``(event_id, entity_id)`` rows for scope building."""
        ...

    async def get_event_entity_pairs_by_entities(
        self,
        entity_ids: list[str],
        source_config_ids: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Return canonical ``(event_id, entity_id)`` rows for entity recall."""
        ...

    async def get_chunks_by_event_ids(
        self,
        event_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Hydrate event chunks while preserving the SAG2 result shape."""
        ...

    async def close(self) -> None:
        """Close resources owned by this store."""
        ...


class ChunkTextSearchStore(Protocol):
    """Keyword retrieval boundary for source chunks.

    The strategy layer consumes this narrow port so the concrete full-text
    engine and its repository implementation remain storage concerns.
    """

    backend_name: str

    async def search_chunks_by_text(
        self,
        *,
        query: str,
        source_config_ids: list[str] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Return chunk payloads ranked by backend text relevance."""
        ...

    async def close(self) -> None:
        """Close resources used by the chunk text backend."""
        ...


class VectorSearchStore(Protocol):
    """Vector/full-text search backend."""

    backend_name: str

    async def health_check(self) -> bool:
        """Return True when the vector/search backend is reachable."""
        ...

    def capabilities(self, database_backend: str) -> StorageCapabilities:
        """Return composed storage capabilities."""
        ...

    async def close(self) -> None:
        """Close vector/search resources owned by this store."""
        ...

    async def upsert_chunk_vectors(
        self, documents: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Upsert chunk vector/search documents."""
        ...

    async def upsert_event_vectors(
        self, documents: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Upsert event vector/search documents."""
        ...

    async def upsert_entity_vectors(
        self, documents: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Upsert entity vector/search documents."""
        ...

    async def upsert_event_entity_vectors(
        self, documents: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Upsert event-entity relation vector/search documents."""
        ...

    async def search_chunks_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Search chunks by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_events_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Search events by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_entities_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Search entities by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_event_entities_by_vector(
        self, *_args: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search event-entity relation vectors."""
        ...


class SearchStore(Protocol):
    """Higher-level retrieval API used by multi-stage searchers."""

    backend_name: str

    async def close(self) -> None:
        """Close resources owned by this store."""
        ...

    async def search_entities_by_text(
        self,
        query: str,
        source_config_ids: list[str] | None = None,
        size: int = 20,
        entity_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search entities by text relevance."""
        ...

    async def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch entity fields by id."""
        ...

    async def get_event_ids_by_entity_ids(
        self,
        *,
        entity_ids: list[str],
        source_config_ids: list[str] | None = None,
        exclude_event_ids: list[str] | None = None,
        size: int = 100,
    ) -> list[str]:
        """Return event ids related to any of the given entity ids."""
        ...

    async def get_events_by_ids(
        self,
        event_ids: list[str],
        source_includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch event fields by id."""
        ...

    async def search_events_by_text(
        self,
        *,
        query: str,
        event_ids: list[str],
        k: int = 100,
        source_config_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank a bounded event candidate set by text relevance."""
        ...

    async def search_events_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Search events by vector."""
        ...

    async def search_chunks_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Search chunks by vector."""
        ...
