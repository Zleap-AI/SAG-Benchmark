"""ES 索引维度隔离命名与 mapping 维度改写。

背景：dense_vector 的 dims 是 mapping 级不可变属性，同一索引不能混存
1024 维与 4096 维向量。因此按维度做物理索引隔离；对上层（脚本/用户）
则通过 source_info.json 记录的维度自动路由，用户永不手写维度。
"""

from __future__ import annotations

import copy
from typing import Any

from pipeline.core.config import get_settings
from pipeline.exceptions import StorageError
from pipeline.utils import get_logger

logger = get_logger("storage.index_naming")

# 无维度后缀索引沿用的维度（见 es_index_legacy_unsuffixed）
LEGACY_DIM = 1024

BASE_INDEX_ENTITY_VECTORS = "entity_vectors"
BASE_INDEX_EVENT_VECTORS = "event_vectors"
BASE_INDEX_EVENT_ENTITY_VECTORS = "event_entity_vectors"
BASE_INDEX_SOURCE_CHUNKS = "source_chunks"

ALL_BASE_INDICES = (
    BASE_INDEX_ENTITY_VECTORS,
    BASE_INDEX_EVENT_VECTORS,
    BASE_INDEX_EVENT_ENTITY_VECTORS,
    BASE_INDEX_SOURCE_CHUNKS,
)


class IndexDimMismatchError(StorageError):
    """已存在索引的 dense_vector dims 与期望维度不符"""


class IndexNotReadyError(StorageError):
    """目标索引缺失，或存在但 mapping 不含期望的 dense_vector 字段。

    后者通常意味着索引被 ES 的 auto_create_index 自动建成了"野生索引"
    （vector 字段被动态映射成普通 float 数组而非 dense_vector）。
    """


def index_suffix(dim: int) -> str:
    """维度后缀。1024 且开启 legacy 兼容时返回空串（沿用现有索引名）。"""
    if dim == LEGACY_DIM and get_settings().es_index_legacy_unsuffixed:
        return ""
    return f"_{dim}"


def resolve_index_name(base_name: str, dim: int | None) -> str:
    """base 索引名 + 维度 → 实际索引名。dim=None 视为 legacy（向后兼容旧调用）。"""
    if dim is None:
        return base_name
    return f"{base_name}{index_suffix(dim)}"


def resolve_all_index_names(dim: int) -> dict[str, str]:
    """{base_name: actual_name}，用于写入 source_info.json 留痕。"""
    return {base: resolve_index_name(base, dim) for base in ALL_BASE_INDICES}


def with_dense_vector_dims(mapping: dict[str, Any], dim: int) -> dict[str, Any]:
    """深拷贝 mapping 并把所有 dense_vector 字段的 dims 改为 dim。

    copy.deepcopy(Doc._doc_type.mapping.to_dict()) 后改 properties.*.dims
    生效且不污染 Document 类本身的 mapping（同一进程内可反复用不同维度）。
    """
    result = copy.deepcopy(mapping)
    changed = []
    for field_name, prop in result.get("properties", {}).items():
        if isinstance(prop, dict) and prop.get("type") == "dense_vector":
            prop["dims"] = dim
            changed.append(field_name)
    if changed:
        logger.debug(f"mapping dense_vector dims -> {dim}: {changed}")
    return result


def extract_dense_vector_dims(mapping: dict[str, Any]) -> dict[str, int]:
    """从 ES 返回的 mapping 中抽出 {field: dims}，用于一致性校验。"""
    return {
        name: prop["dims"]
        for name, prop in (mapping.get("properties") or {}).items()
        if isinstance(prop, dict) and prop.get("type") == "dense_vector" and "dims" in prop
    }


async def assert_index_dims(
    es_client: Any,
    index_name: str,
    expected_dim: int,
    *,
    mapping: dict[str, Any] | None = None,
) -> None:
    """校验已存在索引的 dense_vector dims == expected_dim，否则 fail-fast。

    ES 不允许对已存在索引 put_mapping 修改 dims，也无法把 1024 维向量
    "转换"成 4096 维 —— 只能重新 embed。所以这里只报错，不自动修复。

    传入 mapping 可省去一次 get_mapping 往返（调用方已拉过 mapping 时复用）。
    """
    if mapping is None:
        mapping = await es_client.get_mapping(index=index_name)
    dims = extract_dense_vector_dims(mapping)
    bad = {f: d for f, d in dims.items() if d != expected_dim}
    if bad:
        raise IndexDimMismatchError(
            f"索引 {index_name} 的向量维度与期望不符: {bad} != {expected_dim}。\n"
            f"ES 的 dense_vector.dims 不可变更。请二选一：\n"
            f"  1) 用带维度后缀的新索引（把 ES_INDEX_LEGACY_UNSUFFIXED=false 写进 .env 后重跑 upload）\n"
            f"  2) 手动删除该索引后重跑 upload："
            f' curl -XDELETE "$ES_HOST:$ES_PORT/{index_name}"'
        )
