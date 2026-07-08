"""Protocols for database and vector/search backends."""

from typing import Any, Dict, List, Optional, Protocol

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

    async def upsert_chunk_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Upsert chunk vector/search documents."""
        ...

    async def upsert_event_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Upsert event vector/search documents."""
        ...

    async def upsert_entity_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Upsert entity vector/search documents."""
        ...

    async def upsert_event_entity_vectors(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Upsert event-entity relation vector/search documents."""
        ...

    async def search_chunks_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        """Search chunks by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_events_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        """Search events by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_entities_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        """Search entities by vector. Concrete providers can implement richer signatures."""
        ...

    async def search_event_entities_by_vector(
        self, *_args: Any, **_kwargs: Any
    ) -> List[Dict[str, Any]]:
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
        source_config_ids: Optional[List[str]] = None,
        size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search entities by text relevance."""
        ...

    async def get_event_ids_by_entity_ids(
        self,
        *,
        entity_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
        exclude_event_ids: Optional[List[str]] = None,
        size: int = 100,
    ) -> List[str]:
        """Return event ids related to any of the given entity ids."""
        ...

    async def get_events_by_ids(
        self,
        event_ids: List[str],
        source_includes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch event fields by id."""
        ...

    async def search_events_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        """Search events by vector."""
        ...

    async def search_chunks_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        """Search chunks by vector."""
        ...
