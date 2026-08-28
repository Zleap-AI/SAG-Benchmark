"""Repository 维度路由回归测试（P4：绑定时机统一）。

回归对象：BaseRepository 曾在 __init__ 时固化 INDEX_NAME，导致经 facade 单例
急切构造的 repository 永久锁死在 legacy 无后缀索引上，与 vector 路径读的 _4096
后缀索引脑裂。修复后 INDEX_NAME 改为每次读取时解析 active dim。

不依赖真实 ES：构造 repository 只需一个假的 es_client（EntityVectorRepository 的
get_session_factory 会被 monkeypatch 掉，避免 DB 连接）。
"""

import pytest

from pipeline.storage.backends.elasticsearch.active_dim import (
    reset_active_embedding_dim,
    set_active_embedding_dim,
)
from pipeline.storage.backends.elasticsearch.repositories.entity_repository import (
    EntityVectorRepository,
)
from pipeline.storage.backends.elasticsearch.repositories.event_entity_repository import (
    EventEntityRepository,
)
from pipeline.storage.backends.elasticsearch.repositories.event_repository import (
    EventVectorRepository,
)
from pipeline.storage.backends.elasticsearch.repositories.source_chunk_repository import (
    SourceChunkRepository,
)

_REPOSITORIES = [
    (SourceChunkRepository, "source_chunks"),
    (EventVectorRepository, "event_vectors"),
    (EntityVectorRepository, "entity_vectors"),
    (EventEntityRepository, "event_entity_vectors"),
]


@pytest.fixture(autouse=True)
def _isolate_active_dim():
    reset_active_embedding_dim()
    yield
    reset_active_embedding_dim()


@pytest.fixture
def make_repo(monkeypatch):
    def _make(cls, **kwargs):
        # EntityVectorRepository.__init__ 会调 get_session_factory，避免真连 DB
        if cls is EntityVectorRepository:
            import pipeline.storage.backends.elasticsearch.repositories.entity_repository as m

            monkeypatch.setattr(m, "get_session_factory", lambda: object())
        return cls("es-client", **kwargs)

    return _make


@pytest.mark.parametrize("cls,base_name", _REPOSITORIES)
def test_repo_constructed_before_active_dim_follows_later_dim(cls, base_name, make_repo):
    """核心回归：构造时 active dim 未设，之后才 set → INDEX_NAME 仍应跟随。"""
    reset_active_embedding_dim()
    repo = make_repo(cls)
    set_active_embedding_dim(4096)
    assert repo.INDEX_NAME == f"{base_name}_4096"


@pytest.mark.parametrize("cls,base_name", _REPOSITORIES)
def test_explicit_embedding_dim_overrides_active_dim(cls, base_name, make_repo):
    """显式传 dim 会钉死维度，不跟随 active dim。"""
    set_active_embedding_dim(4096)
    repo = make_repo(cls, embedding_dim=1024)
    # 1024 + 默认 legacy unsuffixed → 无后缀
    assert repo.INDEX_NAME == base_name


@pytest.mark.parametrize("cls,base_name", _REPOSITORIES)
def test_no_active_dim_falls_back_to_legacy_unsuffixed(cls, base_name, make_repo):
    reset_active_embedding_dim()
    repo = make_repo(cls)
    assert repo.INDEX_NAME == base_name


def test_active_dim_change_takes_effect_immediately(make_repo):
    repo = make_repo(SourceChunkRepository)
    set_active_embedding_dim(4096)
    assert repo.INDEX_NAME == "source_chunks_4096"
    reset_active_embedding_dim()
    assert repo.INDEX_NAME == "source_chunks"


@pytest.mark.parametrize("cls,_base_name", _REPOSITORIES)
def test_subclasses_do_not_shadow_index_name(cls, _base_name):
    """守卫：子类不得重新定义类级 INDEX_NAME，否则会遮蔽基类 property。"""
    assert "INDEX_NAME" not in cls.__dict__
