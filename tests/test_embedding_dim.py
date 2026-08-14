"""embedding 维度动态解析 + 索引命名的单元测试（不依赖 ES / embedding 服务）"""

import asyncio

import pytest

from pipeline.core.ai import embedding_dim as ed
from pipeline.storage import indexing as inaming


class _FakeClient:
    def __init__(self, dim: int):
        self.dim = dim
        self.calls = 0

    async def probe_dimensions(self, probe_text: str = "dimension probe") -> int:
        self.calls += 1
        return self.dim


class _BoomClient:
    async def probe_dimensions(self, probe_text: str = "dimension probe") -> int:
        raise ConnectionError("endpoint down")


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    ed.reset_embedding_dim_cache()
    monkeypatch.setattr(ed, "_cache_path", lambda: tmp_path / "embedding_dims.json")
    yield
    ed.reset_embedding_dim_cache()


def test_index_suffix_legacy_unsuffixed():
    # 默认 ES_INDEX_LEGACY_UNSUFFIXED=True → 1024 维沿用现有无后缀索引名
    assert inaming.resolve_index_name("entity_vectors", 1024) == "entity_vectors"
    assert inaming.resolve_index_name("entity_vectors", 4096) == "entity_vectors_4096"
    assert inaming.resolve_index_name("entity_vectors", None) == "entity_vectors"


def test_index_suffix_non_legacy():
    assert inaming.index_suffix(1024) == ""
    assert inaming.index_suffix(4096) == "_4096"
    assert inaming.index_suffix(768) == "_768"


def test_with_dense_vector_dims_does_not_mutate_source():
    base = {
        "properties": {
            "content_vector": {"type": "dense_vector", "dims": 1024, "index": True},
            "heading": {"type": "text"},
        }
    }
    out = inaming.with_dense_vector_dims(base, 4096)
    assert out["properties"]["content_vector"]["dims"] == 4096
    assert base["properties"]["content_vector"]["dims"] == 1024  # 原对象未被污染
    assert out["properties"]["heading"] == {"type": "text"}


def test_with_dense_vector_dims_non_vector_fields_unchanged():
    base = {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "vec": {"type": "dense_vector", "dims": 1024},
            "text": {"type": "text"},
        }
    }
    out = inaming.with_dense_vector_dims(base, 768)
    assert out["properties"]["vec"]["dims"] == 768
    assert out["properties"]["chunk_id"] == {"type": "keyword"}
    assert base["properties"]["vec"]["dims"] == 1024


def test_extract_dense_vector_dims():
    mapping = {
        "properties": {
            "vec_a": {"type": "dense_vector", "dims": 1024},
            "vec_b": {"type": "dense_vector", "dims": 4096},
            "text": {"type": "text"},
            "vec_no_dims": {"type": "dense_vector"},
        }
    }
    result = inaming.extract_dense_vector_dims(mapping)
    assert result == {"vec_a": 1024, "vec_b": 4096}


def test_cache_key_isolation():
    k1 = ed.make_cache_key("http://a/v1", "m1", None)
    k2 = ed.make_cache_key("http://b/v1", "m1", None)
    k3 = ed.make_cache_key("http://a/v1", "m2", None)
    k4 = ed.make_cache_key("http://a/v1", "m1", 512)
    assert len({k1, k2, k3, k4}) == 4


def test_cache_key_with_none_base_url():
    key = ed.make_cache_key(None, "m1", None)
    assert isinstance(key, str) and len(key) == 16


def test_probe_then_cache_hits(monkeypatch):
    client = _FakeClient(4096)
    r1 = asyncio.run(ed.resolve_embedding_dim(client=client))
    assert r1["dim"] == 4096 and r1["dim_source"] == "probe" and client.calls == 1
    # L1 命中，不再 probe
    r2 = asyncio.run(ed.resolve_embedding_dim(client=client))
    assert r2["dim_source"] == "cache" and client.calls == 1
    # 落盘存在且可读（清 L1 后从 L2 读回）
    ed.reset_embedding_dim_cache()
    r3 = asyncio.run(ed.resolve_embedding_dim(client=client))
    assert r3["dim"] == 4096 and r3["dim_source"] == "cache" and client.calls == 1


def test_probe_failure_falls_back_to_env():
    from pipeline.core.config import get_settings

    settings = get_settings()
    assert settings.embedding_dimensions == 1024, "本测试假设 .env 的 EMBEDDING_DIM=1024"
    r = asyncio.run(ed.resolve_embedding_dim(client=_BoomClient()))
    assert r["dim"] == 1024 and r["dim_source"] == "env"


def test_probe_failure_prefers_stale_cache():
    client = _FakeClient(2048)
    asyncio.run(ed.resolve_embedding_dim(client=client))
    ed.reset_embedding_dim_cache()
    r = asyncio.run(ed.resolve_embedding_dim(client=_BoomClient()))
    assert r["dim"] == 2048 and r["dim_source"] in ("cache", "stale_cache")


def test_resolve_all_index_names():
    result = inaming.resolve_all_index_names(4096)
    expected = {
        "entity_vectors": "entity_vectors_4096",
        "event_vectors": "event_vectors_4096",
        "event_entity_vectors": "event_entity_vectors_4096",
        "source_chunks": "source_chunks_4096",
    }
    assert result == expected


def test_resolve_all_index_names_legacy():
    result = inaming.resolve_all_index_names(1024)
    # 1024 with legacy mode → all unsuffixed
    for base, actual in result.items():
        assert base == actual
