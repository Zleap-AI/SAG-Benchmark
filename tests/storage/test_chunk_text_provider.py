"""Contract tests for the source-chunk keyword-search provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.storage.providers import chunk_text as chunk_text_provider


@pytest.mark.asyncio
async def test_elasticsearch_chunk_text_provider_is_lazy_and_delegates(monkeypatch):
    repository = SimpleNamespace(
        search_by_text=AsyncMock(return_value=[{"chunk_id": "chunk-1", "_score": 2.5}])
    )
    get_client = MagicMock()
    repository_factory_calls = []

    def _get_client():
        get_client()
        return "es-client"

    def _repository_factory(client):
        repository_factory_calls.append(client)
        return repository

    monkeypatch.setattr(chunk_text_provider, "get_es_client", _get_client)
    monkeypatch.setattr(chunk_text_provider, "SourceChunkRepository", _repository_factory)

    store = chunk_text_provider.ElasticsearchChunkTextSearchStore()
    assert get_client.call_count == 0

    result = await store.search_chunks_by_text(
        query="event universe",
        source_config_ids=["source-a", "source-b"],
        size=7,
    )

    assert result == [{"chunk_id": "chunk-1", "_score": 2.5}]
    assert get_client.call_count == 1
    assert repository_factory_calls == ["es-client"]
    repository.search_by_text.assert_awaited_once_with(
        query="event universe",
        source_config_ids=["source-a", "source-b"],
        size=7,
    )


@pytest.mark.asyncio
async def test_elasticsearch_chunk_text_provider_closes_only_after_use(monkeypatch):
    close_client = AsyncMock()
    monkeypatch.setattr(chunk_text_provider, "close_es_client", close_client)
    monkeypatch.setattr(chunk_text_provider, "get_es_client", lambda: "es-client")
    monkeypatch.setattr(
        chunk_text_provider,
        "SourceChunkRepository",
        lambda _client: SimpleNamespace(search_by_text=AsyncMock(return_value=[])),
    )

    unused_store = chunk_text_provider.ElasticsearchChunkTextSearchStore()
    await unused_store.close()
    close_client.assert_not_awaited()

    used_store = chunk_text_provider.ElasticsearchChunkTextSearchStore()
    await used_store.search_chunks_by_text(query="query")
    await used_store.close()
    close_client.assert_awaited_once_with()
