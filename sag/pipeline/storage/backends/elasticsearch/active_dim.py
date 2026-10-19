"""进程级 "当前活跃 embedding 维度"。

Elasticsearch Provider 下的 Repository 共享同一进程级索引路由上下文。
脚本入口通过 pipeline.storage.indexing 设置一次，所有 Repository 自动继承。
显式传 embedding_dim 的调用点优先级更高。
"""

from __future__ import annotations

from pipeline.utils import get_logger

logger = get_logger("storage.active_dim")

_ACTIVE_DIM: int | None = None


def set_active_embedding_dim(dim: int | None) -> None:
    global _ACTIVE_DIM
    if _ACTIVE_DIM is not None and dim is not None and _ACTIVE_DIM != dim:
        logger.warning(f"活跃 embedding 维度变更: {_ACTIVE_DIM} -> {dim}")
    _ACTIVE_DIM = dim
    logger.info(f"活跃 embedding 维度设为: {dim}")


def get_active_embedding_dim() -> int | None:
    """未设置时返回 None → resolve_index_name 退化为 legacy 无后缀索引名。"""
    return _ACTIVE_DIM


def reset_active_embedding_dim() -> None:
    global _ACTIVE_DIM
    _ACTIVE_DIM = None
