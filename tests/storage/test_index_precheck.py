"""P3 索引预检单测：assert_indices_ready 必须在写入前 fail-fast。

不依赖真实 ES：注入一个 fake es_client（index_exists / get_mapping），验证
缺失索引、野生索引（mapping 缺 dense_vector 字段）、dims 不符三种失败模式，
以及全部就绪的通过模式。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipeline.storage import indexing as inaming
from pipeline.storage.backends.elasticsearch.index_naming import (
    IndexDimMismatchError,
    IndexNotReadyError,
)


def _mapping(*dense_vector_fields: str, dims: int = 4096) -> dict:
    return {
        "properties": {
            **{f: {"type": "dense_vector", "dims": dims} for f in dense_vector_fields},
            "id": {"type": "keyword"},
        }
    }


# 期望字段：从 Document 类反推。这里按实际 mapping 硬编码，作为预检契约的独立参照。
_EXPECTED_FIELDS = {
    "source_chunks": ["content_vector", "heading_vector"],
    "event_vectors": ["content_vector", "title_vector"],
    "entity_vectors": ["vector"],
    "event_entity_vectors": ["vector"],
}


def _fake_client(*, existing: dict[str, dict] | None = None) -> SimpleNamespace:
    """existing: {index_name: mapping}。缺省映射的索引视为存在但 wild（无 dense_vector）。"""
    existing = existing or {}

    async def _index_exists(name):
        return name in existing

    async def _get_mapping(index, **kwargs):
        if index not in existing:
            return {}
        return existing[index]

    return SimpleNamespace(index_exists=_index_exists, get_mapping=_get_mapping)


def _all_ready_mappings() -> dict[str, dict]:
    return {
        "source_chunks_4096": _mapping("content_vector", "heading_vector"),
        "event_vectors_4096": _mapping("content_vector", "title_vector"),
        "entity_vectors_4096": _mapping("vector"),
        "event_entity_vectors_4096": _mapping("vector"),
    }


@pytest.mark.asyncio
async def test_all_ready_returns_resolved_names():
    client = _fake_client(existing=_all_ready_mappings())
    result = await inaming.assert_indices_ready(4096, es_client=client)
    assert result == inaming.resolve_all_index_names(4096)


@pytest.mark.asyncio
async def test_missing_index_raises_with_hint():
    client = _fake_client(existing={})  # 全部缺失
    with pytest.raises(IndexNotReadyError) as exc:
        await inaming.assert_indices_ready(4096, es_client=client)
    msg = str(exc.value)
    assert "init_elasticsearch.py" in msg
    assert "source_chunks_4096" in msg


@pytest.mark.asyncio
async def test_wild_index_missing_dense_vector_raises():
    """野生索引：索引存在但 content_vector 是普通 float，缺 dense_vector 字段。"""
    wild = {"properties": {"content_vector": {"type": "float"}}}
    existing = _all_ready_mappings()
    existing["source_chunks_4096"] = wild  # 用野生索引覆盖 source_chunks
    client = _fake_client(existing=existing)
    with pytest.raises(IndexNotReadyError) as exc:
        await inaming.assert_indices_ready(4096, es_client=client)
    assert "source_chunks_4096" in str(exc.value)


@pytest.mark.asyncio
async def test_dims_mismatch_raises_index_dim_mismatch():
    wrong_dims = _mapping("content_vector", "heading_vector", dims=1024)
    existing = _all_ready_mappings()
    existing["source_chunks_4096"] = wrong_dims
    client = _fake_client(existing=existing)
    with pytest.raises(IndexDimMismatchError):
        await inaming.assert_indices_ready(4096, es_client=client)


@pytest.mark.asyncio
async def test_missing_and_dims_mismatch_both_reported():
    """缺失索引与 dims 不符同时存在时，报错须同时列出两者（不能只报 dims 不符漏掉缺失）。"""
    existing = _all_ready_mappings()
    del existing["entity_vectors_4096"]  # 缺失
    existing["event_vectors_4096"] = _mapping(  # dims 不符
        "content_vector", "title_vector", dims=1024
    )
    client = _fake_client(existing=existing)
    with pytest.raises(IndexNotReadyError) as exc:
        await inaming.assert_indices_ready(4096, es_client=client)
    msg = str(exc.value)
    assert "entity_vectors_4096" in msg  # 缺失索引必须被列出
    assert "event_vectors_4096" in msg  # dims 不符也必须被列出


@pytest.mark.asyncio
async def test_assert_index_dims_reuses_provided_mapping():
    """传入 mapping 时不再调用 get_mapping（省一次往返）。"""
    get_mapping = AsyncMock(return_value=_mapping("content_vector"))
    client = SimpleNamespace(get_mapping=get_mapping)
    await inaming.assert_index_dims(
        client, "source_chunks_4096", 4096, mapping=_mapping("content_vector")
    )
    get_mapping.assert_not_awaited()


@pytest.mark.asyncio
async def test_1024_legacy_checks_unsuffixed_names():
    """1024 + 默认 legacy unsuffixed → 预检无后缀索引名。"""
    existing = {
        "source_chunks": _mapping("content_vector", "heading_vector", dims=1024),
        "event_vectors": _mapping("content_vector", "title_vector", dims=1024),
        "entity_vectors": _mapping("vector", dims=1024),
        "event_entity_vectors": _mapping("vector", dims=1024),
    }
    client = _fake_client(existing=existing)
    result = await inaming.assert_indices_ready(1024, es_client=client)
    assert result == {
        "source_chunks": "source_chunks",
        "event_vectors": "event_vectors",
        "entity_vectors": "entity_vectors",
        "event_entity_vectors": "event_entity_vectors",
    }
