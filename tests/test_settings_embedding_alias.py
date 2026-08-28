"""配置断链修复验证。测试不依赖开发机 .env。"""

import pytest

from pipeline.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_embedding_dim_alias_reads_env_dim():
    """EMBEDDING_DIM=1024（.env）必须被 embedding_dimensions 读到 —— 这是本次修复的核心断链"""
    assert get_settings().embedding_dimensions == 1024


def test_new_config_fields_have_defaults():
    """新增字段应有合理的默认值"""
    s = get_settings()
    assert s.embedding_request_dimensions is None
    assert s.embedding_dim_strict is False
    assert s.embedding_dim_probe_timeout == 30
    assert s.es_index_legacy_unsuffixed is True
