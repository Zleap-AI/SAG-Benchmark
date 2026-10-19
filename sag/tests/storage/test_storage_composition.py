"""Composition and lifecycle tests for the unified storage facade."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.storage import factory
from pipeline.storage.facade import StorageFacade


@pytest.mark.asyncio
async def test_facade_closes_chunk_text_provider_with_other_stores():
    database = SimpleNamespace(close=AsyncMock())
    vector = SimpleNamespace(close=AsyncMock())
    search = SimpleNamespace(close=AsyncMock())
    chunk_text = SimpleNamespace(close=AsyncMock())
    facade = StorageFacade(
        database=database,
        vector=vector,
        search=search,
        chunk_text=chunk_text,
    )

    await facade.close()

    chunk_text.close.assert_awaited_once_with()
    vector.close.assert_awaited_once_with()
    search.close.assert_awaited_once_with()
    database.close.assert_awaited_once_with()


def test_factory_composes_lazy_elasticsearch_chunk_text_provider(monkeypatch):
    database = SimpleNamespace(backend_name="oceanbase")
    vector = SimpleNamespace(backend_name="oceanbase")
    search = SimpleNamespace(backend_name="oceanbase")
    chunk_text = SimpleNamespace(backend_name="elasticsearch")

    monkeypatch.setattr(factory, "OceanBaseDatabaseStore", MagicMock(return_value=database))
    monkeypatch.setattr(factory, "OceanBaseVectorStore", MagicMock(return_value=vector))
    monkeypatch.setattr(factory, "OceanBaseSearchStore", MagicMock(return_value=search))
    chunk_text_factory = MagicMock(return_value=chunk_text)
    monkeypatch.setattr(
        factory,
        "ElasticsearchChunkTextSearchStore",
        chunk_text_factory,
    )

    facade = factory.create_storage_facade(
        SimpleNamespace(
            effective_database_backend="oceanbase",
            effective_vector_backend="oceanbase",
        )
    )

    chunk_text_factory.assert_called_once_with()
    assert facade.database is database
    assert facade.vector is vector
    assert facade.search is search
    assert facade.event_universe is database
    assert facade.chunk_text is chunk_text


def test_facade_keeps_manual_pre_migration_composition_compatible():
    database = SimpleNamespace()
    facade = StorageFacade(
        database=database,
        vector=SimpleNamespace(),
        search=SimpleNamespace(),
    )

    assert facade.event_universe is database
    assert facade.chunk_text is None
