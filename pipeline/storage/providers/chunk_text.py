"""Source-chunk keyword-search providers."""

from typing import Any, Dict, List, Optional

from pipeline.storage.backends.elasticsearch.client import close_es_client, get_es_client
from pipeline.storage.backends.elasticsearch.repositories.source_chunk_repository import SourceChunkRepository


class ElasticsearchChunkTextSearchStore:
    """Elasticsearch BM25 adapter for source chunks.

    Repository construction is lazy so non-BM25 strategies do not initialize
    Elasticsearch solely because the composed storage facade was created.
    """

    backend_name = "elasticsearch"

    def __init__(self) -> None:
        self._repository: Optional[SourceChunkRepository] = None

    def _get_repository(self) -> SourceChunkRepository:
        if self._repository is None:
            self._repository = SourceChunkRepository(get_es_client())
        return self._repository

    async def search_chunks_by_text(
        self,
        *,
        query: str,
        source_config_ids: Optional[List[str]] = None,
        size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Delegate to the existing ES repository without changing scoring."""
        return await self._get_repository().search_by_text(
            query=query,
            source_config_ids=source_config_ids,
            size=size,
        )

    async def close(self) -> None:
        """Release the shared ES client only when this provider initialized it."""
        if self._repository is not None:
            self._repository = None
            await close_es_client()


__all__ = ["ElasticsearchChunkTextSearchStore"]
