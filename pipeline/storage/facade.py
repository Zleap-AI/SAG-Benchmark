"""Composed storage facade."""

from typing import Optional

from pipeline.core.config import get_settings
from pipeline.storage.capabilities import StorageHealth
from pipeline.storage.interfaces import DatabaseStore, SearchStore, VectorSearchStore


class StorageFacade:
    """Facade that composes a structured database backend and a vector backend."""

    def __init__(
        self,
        database: DatabaseStore,
        vector: VectorSearchStore,
        search: SearchStore,
    ) -> None:
        self.database = database
        self.vector = vector
        self.search = search

    async def health_check(self) -> StorageHealth:
        """Check both configured storage backends."""
        database_ok = await self.database.health_check()
        vector_ok = await self.vector.health_check()
        capabilities = self.vector.capabilities(self.database.backend_name)
        ok = database_ok and vector_ok
        return StorageHealth(
            ok=ok,
            database_ok=database_ok,
            vector_ok=vector_ok,
            capabilities=capabilities,
            message="ok" if ok else "one or more storage backends are unavailable",
        )

    async def close(self) -> None:
        """Close both stores."""
        await self.vector.close()
        await self.search.close()
        await self.database.close()


_storage_facade: Optional[StorageFacade] = None


def get_storage_facade() -> StorageFacade:
    """Return the process-wide storage facade."""
    global _storage_facade
    if _storage_facade is None:
        from pipeline.storage.provider import create_storage_facade

        _storage_facade = create_storage_facade(get_settings())
    return _storage_facade


async def close_storage_facade() -> None:
    """Close and reset the process-wide storage facade.

    Safe to call even when the facade was never created.  After this call
    the cached singleton is dropped so the next ``get_storage_facade()``
    creates a fresh instance (useful for tests that reconfigure backends).
    """
    global _storage_facade
    if _storage_facade is not None:
        await _storage_facade.close()
        _storage_facade = None


def reset_storage_facade() -> None:
    """Drop the cached facade without closing it.

    Prefer :func:`close_storage_facade` unless you are certain the
    underlying stores have already been shut down separately.
    """
    global _storage_facade
    _storage_facade = None
