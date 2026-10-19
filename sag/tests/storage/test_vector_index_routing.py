"""维度路由契约测试：ElasticsearchVectorStore 必须按 active dim 解析索引名。

回归对象：vector.py 曾硬编码无后缀索引名（source_chunks 等），导致 4096 维
向量写入 1024 维旧索引全部失败。这里不依赖真实 ES，monkeypatch get_es_client
为 mock，断言 4 个 upsert + 4 个 search 传给 client 的 index 都带正确后缀。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.storage.backends.elasticsearch.active_dim import (
    reset_active_embedding_dim,
    set_active_embedding_dim,
)
from pipeline.storage.providers import vector as vector_provider

_ES_BULK_RESULT = {
    "success_count": 1,
    "error_count": 0,
    "errors": [],
}

_UPSERT_CASES = [
    ("upsert_chunk_vectors", "source_chunks"),
    ("upsert_event_vectors", "event_vectors"),
    ("upsert_entity_vectors", "entity_vectors"),
    ("upsert_event_entity_vectors", "event_entity_vectors"),
]

_SEARCH_CASES = [
    ("search_chunks_by_vector", "source_chunks"),
    ("search_events_by_vector", "event_vectors"),
    ("search_entities_by_vector", "entity_vectors"),
    ("search_event_entities_by_vector", "event_entity_vectors"),
]


@pytest.fixture(autouse=True)
def _isolate_active_dim():
    reset_active_embedding_dim()
    yield
    reset_active_embedding_dim()


def _make_store() -> tuple:
    bulk_index = AsyncMock(return_value=_ES_BULK_RESULT)
    vector_search = AsyncMock(return_value=[])
    es_client = SimpleNamespace(bulk_index=bulk_index, vector_search=vector_search)
    get_client_stub = MagicMock(return_value=es_client)
    store = vector_provider.ElasticsearchVectorStore()
    return store, es_client, get_client_stub


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name,base_name", _UPSERT_CASES)
async def test_upsert_routes_to_suffixed_index_under_active_dim(
    monkeypatch, method_name, base_name
):
    store, es_client, get_client = _make_store()
    monkeypatch.setattr(vector_provider, "get_es_client", get_client)
    set_active_embedding_dim(4096)
    await getattr(store, method_name)([{"id": "d1", "vector": [0.1] * 4096}])

    assert es_client.bulk_index.await_count == 1
    assert es_client.bulk_index.await_args.kwargs["index"] == f"{base_name}_4096"


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name,base_name", _SEARCH_CASES)
async def test_search_routes_to_suffixed_index_under_active_dim(
    monkeypatch, method_name, base_name
):
    store, es_client, get_client = _make_store()
    monkeypatch.setattr(vector_provider, "get_es_client", get_client)
    set_active_embedding_dim(4096)
    await getattr(store, method_name)([0.1] * 4096, k=3)

    assert es_client.vector_search.await_count == 1
    assert es_client.vector_search.await_args.kwargs["index"] == f"{base_name}_4096"


@pytest.mark.asyncio
async def test_legacy_unsuffixed_fallback_when_no_active_dim(monkeypatch):
    """未设置 active dim 时退化为无后缀索引名（legacy 兼容回归）。"""
    store, es_client, get_client = _make_store()
    monkeypatch.setattr(vector_provider, "get_es_client", get_client)

    await store.upsert_chunk_vectors([{"id": "d1", "vector": [0.1] * 1024}])
    await store.search_chunks_by_vector([0.1] * 1024, k=1)

    assert es_client.bulk_index.await_args.kwargs["index"] == "source_chunks"
    assert es_client.vector_search.await_args.kwargs["index"] == "source_chunks"


@pytest.mark.asyncio
async def test_1024_legacy_unsuffixed_production_path(monkeypatch):
    """1024 维 + 默认 ES_INDEX_LEGACY_UNSUFFIXED=True → 仍写无后缀索引。

    这是现存 1024 维数据的生产主路径，走 index_suffix 里读 settings 的
    分支（区别于 dim=None 分支），必须独立覆盖。
    """
    store, es_client, get_client = _make_store()
    monkeypatch.setattr(vector_provider, "get_es_client", get_client)
    set_active_embedding_dim(1024)
    await store.upsert_chunk_vectors([{"id": "d1", "vector": [0.1] * 1024}])

    assert es_client.bulk_index.await_args.kwargs["index"] == "source_chunks"


def test_index_resolves_every_call_not_cached():
    """_index 每次调用重新解析，不缓存 —— active dim 变更后立即生效。"""
    store = vector_provider.ElasticsearchVectorStore()
    set_active_embedding_dim(4096)
    assert store._index("source_chunks") == "source_chunks_4096"
    reset_active_embedding_dim()
    assert store._index("source_chunks") == "source_chunks"
