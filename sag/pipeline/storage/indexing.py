"""Public storage index-routing API.

Upload and benchmark entry points use this module to coordinate the active
embedding dimension without depending on Elasticsearch backend internals.
"""

from typing import Any

from pipeline.storage.backends.elasticsearch.active_dim import (
    get_active_embedding_dim,
    reset_active_embedding_dim,
    set_active_embedding_dim,
)
from pipeline.storage.backends.elasticsearch.index_naming import (
    ALL_BASE_INDICES,
    BASE_INDEX_ENTITY_VECTORS,
    BASE_INDEX_EVENT_ENTITY_VECTORS,
    BASE_INDEX_EVENT_VECTORS,
    BASE_INDEX_SOURCE_CHUNKS,
    LEGACY_DIM,
    IndexDimMismatchError,
    IndexNotReadyError,
    assert_index_dims,
    extract_dense_vector_dims,
    index_suffix,
    resolve_all_index_names,
    resolve_index_name,
    with_dense_vector_dims,
)
from pipeline.utils import get_logger

logger = get_logger("storage.indexing")

__all__ = [
    "ALL_BASE_INDICES",
    "BASE_INDEX_ENTITY_VECTORS",
    "BASE_INDEX_EVENT_ENTITY_VECTORS",
    "BASE_INDEX_EVENT_VECTORS",
    "BASE_INDEX_SOURCE_CHUNKS",
    "LEGACY_DIM",
    "IndexDimMismatchError",
    "IndexNotReadyError",
    "assert_index_dims",
    "assert_indices_ready",
    "extract_dense_vector_dims",
    "get_active_embedding_dim",
    "index_suffix",
    "reset_active_embedding_dim",
    "resolve_all_index_names",
    "resolve_index_name",
    "set_active_embedding_dim",
    "with_dense_vector_dims",
]


def _format_not_ready(
    dim: int,
    missing: list[str],
    malformed: list[tuple[str, list[str]]],
    dims_mismatch: list[tuple[str, dict[str, int]]] | None = None,
) -> str:
    lines = [f"❌ ES 索引未就绪 (embedding_dim={dim})，已中止写入以避免产生野生索引："]
    if missing:
        lines.append(f"   • 缺失索引: {', '.join(missing)}")
    for name, fields in malformed:
        lines.append(
            f"   • {name}: mapping 缺少 dense_vector 字段 {fields}"
            f"（疑似被 ES auto_create_index 自动创建的野生索引）"
        )
    for name, bad in dims_mismatch or []:
        lines.append(f"   • {name}: dense_vector dims 不符 {bad} != {dim}")
    lines += [
        "",
        "   ES 开启 auto_create_index 时，向不存在的索引写入会被自动建成没有",
        "   dense_vector / routing / is_delete 的野生索引：写入报“成功”，但向量检索静默返回 0 条。",
        "",
        "   请先初始化索引后重跑 upload：",
        f"       uv run python scripts/init_elasticsearch.py --dim {dim}",
    ]
    if malformed or dims_mismatch:
        lines += [
            "   若上述索引已被错误创建，需先删除再初始化：",
            *(f'       curl -XDELETE "$ES_HOST:$ES_PORT/{n}"' for n, _ in malformed),
            *(f'       curl -XDELETE "$ES_HOST:$ES_PORT/{n}"' for n, _ in dims_mismatch or []),
        ]
    return "\n".join(lines)


def _format_dims_mismatch(dim: int, dims_mismatch: list[tuple[str, dict[str, int]]]) -> str:
    lines = [f"❌ ES 索引 dense_vector dims 与 embedding_dim={dim} 不符，已中止写入："]
    for name, bad in dims_mismatch:
        lines.append(f"   • {name}: {bad} != {dim}")
    lines += [
        "",
        "   ES 的 dense_vector.dims 是 mapping 级不可变属性，无法原地修改，只能重新 embed。",
        "   请二选一：",
        "      1) 用带维度后缀的新索引重跑 upload（写入 source_info.json 自动路由）",
        *(
            f'      2) 删除该索引后重跑 init：curl -XDELETE "$ES_HOST:$ES_PORT/{n}"'
            for n, _ in dims_mismatch
        ),
    ]
    return "\n".join(lines)


async def assert_indices_ready(
    dim: int,
    *,
    es_client: Any | None = None,
) -> dict[str, str]:
    """写入前预检：dim 对应的 4 个 ES 索引必须已存在，且 dense_vector 字段齐全、dims 匹配。

    背景：ES 集群 auto_create_index=true 时，向不存在的索引 bulk 写入会被自动
    建成"野生索引"（content_vector 被动态映射成普通 float 数组而非 dense_vector、
    routing/is_delete 字段缺失），写入报"成功"但向量检索静默召回 0 条。

    Args:
        dim: 目标 embedding 维度（通常来自 resolve_embedding_dim()）。
        es_client: 可选注入（测试用）；默认取进程级 ES 单例。

    Returns:
        {base_index_name: actual_index_name}，与 resolve_all_index_names(dim) 同形。

    Raises:
        IndexNotReadyError: 索引缺失或 mapping 异常（野生索引）。
        IndexDimMismatchError: 索引存在但 dense_vector dims 与 dim 不符。
    """
    from pipeline.storage.backends.elasticsearch.client import get_es_client
    from pipeline.storage.backends.elasticsearch.documents import REGISTERED_DOCUMENTS

    client = es_client or get_es_client()
    # 期望的向量字段：从 Document 类 mapping 反推，新增索引时零维护
    expected_fields = {
        getattr(doc, "BASE_INDEX_NAME", doc.Index.name): sorted(
            extract_dense_vector_dims(doc._doc_type.mapping.to_dict())
        )
        for doc in REGISTERED_DOCUMENTS
    }

    names = resolve_all_index_names(dim)
    missing: list[str] = []
    malformed: list[tuple[str, list[str]]] = []
    dims_mismatch: list[tuple[str, dict[str, int]]] = []

    for base, name in names.items():
        if not await client.index_exists(name):
            missing.append(name)
            continue
        mapping = await client.get_mapping(index=name)
        present = extract_dense_vector_dims(mapping)
        absent = [f for f in expected_fields.get(base, []) if f not in present]
        if absent:
            malformed.append((name, absent))
            continue
        bad = {f: d for f, d in present.items() if d != dim}
        if bad:
            dims_mismatch.append((name, bad))

    if missing or malformed:
        raise IndexNotReadyError(_format_not_ready(dim, missing, malformed, dims_mismatch))
    if dims_mismatch:
        raise IndexDimMismatchError(_format_dims_mismatch(dim, dims_mismatch))

    logger.info(f"ES 索引预检通过 (dim={dim}): {list(names.values())}")
    return names
