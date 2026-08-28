"""
Initialize OceanBase full-mode vector columns and vector indexes.

This module is used by scripts/init_database.py when STORAGE_PROFILE resolves
to OceanBase full mode. It can also be run directly for repairing/rebuilding
OceanBase vector columns and indexes. Run it only when:

    DATABASE_BACKEND=oceanbase
    VECTOR_BACKEND=oceanbase
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

from sqlalchemy import text

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.core.config import get_settings
from pipeline.db.base import close_database, get_engine


def print_header(text_: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text_}")
    print("=" * 70)


def print_success(text_: str) -> None:
    print(f"  [OK] {text_}")


def print_info(text_: str) -> None:
    print(f"  - {text_}")


def print_warning(text_: str) -> None:
    print(f"  [WARN] {text_}")


async def column_exists(table_name: str, column_name: str) -> bool:
    settings = get_settings()
    database_name = settings.oceanbase_database
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = :database_name
                  AND table_name = :table_name
                  AND column_name = :column_name
                """),
            {
                "database_name": database_name,
                "table_name": table_name,
                "column_name": column_name,
            },
        )
        return int(result.scalar_one()) > 0


async def get_column_type(table_name: str, column_name: str) -> str:
    """查询 information_schema.columns 返回列的 COLUMN_TYPE（小写规范化）。"""
    settings = get_settings()
    database_name = settings.oceanbase_database
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT COLUMN_TYPE
                FROM information_schema.columns
                WHERE table_schema = :database_name
                  AND table_name = :table_name
                  AND column_name = :column_name
                """),
            {
                "database_name": database_name,
                "table_name": table_name,
                "column_name": column_name,
            },
        )
        row = result.fetchone()
        return row[0].lower().replace(" ", "") if row else ""


def _parse_vector_dim(type_str: str) -> int | None:
    """从 'vector(1024)' 等类型字符串中解析出维度数值；非 vector 类型返回 None。"""
    m = re.fullmatch(r"vector\((\d+)\)", type_str.lower().replace(" ", ""))
    return int(m.group(1)) if m else None


async def add_column_if_missing(
    table_name: str,
    column_name: str,
    column_sql: str,
    expected_type: str | None = None,
) -> None:
    """为表添加列（幂等）。

    若列已存在且 expected_type 不为 None，则校验实际列类型是否与期望一致。
    仅对向量列（VECTOR 类型）传入 expected_type；普通 JSON 列不传。
    维度不符时抛出 RuntimeError，阻止后续步骤静默通过。
    """
    if await column_exists(table_name, column_name):
        if expected_type is not None:
            expected_dim = _parse_vector_dim(expected_type)
            if expected_dim is not None:
                actual_type = await get_column_type(table_name, column_name)
                actual_dim = _parse_vector_dim(actual_type)
                if actual_dim != expected_dim:
                    raise RuntimeError(
                        f"{table_name}.{column_name}: 向量列维度不匹配 "
                        f"（实际类型={actual_type}, 期望类型={expected_type.lower().replace(' ', '')}）"
                        f"——请手动迁移列后重新运行初始化脚本"
                    )
        print_info(f"{table_name}.{column_name}: already exists")
        return

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))
    print_success(f"{table_name}.{column_name}: added")


async def index_exists(table_name: str, index_name: str) -> bool:
    settings = get_settings()
    database_name = settings.oceanbase_database
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT COUNT(*)
                FROM information_schema.statistics
                WHERE table_schema = :database_name
                  AND table_name = :table_name
                  AND index_name = :index_name
                """),
            {
                "database_name": database_name,
                "table_name": table_name,
                "index_name": index_name,
            },
        )
        return int(result.scalar_one()) > 0


def build_vector_index_params() -> str:
    settings = get_settings()
    index_type = settings.oceanbase_vector_index_type.upper()
    distance = "COSINE"
    lib = settings.oceanbase_vector_index_lib.upper()

    params = [
        f"distance={distance}",
        f"type={index_type}",
        f"lib={lib}",
    ]

    if index_type.startswith("HNSW"):
        params.extend(
            [
                f"m={settings.oceanbase_vector_index_m}",
                f"ef_construction={settings.oceanbase_vector_index_ef_construction}",
            ]
        )

    return ", ".join(params)


async def create_vector_index_if_missing(
    table_name: str,
    column_name: str,
    index_name: str,
) -> None:
    if await index_exists(table_name, index_name):
        print_info(f"{table_name}.{index_name}: already exists")
        return

    params = build_vector_index_params()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(f"CREATE VECTOR INDEX {index_name} ON {table_name}({column_name}) WITH ({params})")
        )
    print_success(f"{table_name}.{index_name}: created")


async def drop_index_if_exists(table_name: str, index_name: str) -> None:
    if not await index_exists(table_name, index_name):
        return

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"ALTER TABLE {table_name} DROP INDEX {index_name}"))
    print_success(f"{table_name}.{index_name}: dropped")


async def create_fulltext_index_if_missing(
    table_name: str,
    index_name: str,
    columns: list[str],
    parser: str | None = "ik",
    parser_properties: str | None = 'ik_mode="max_word"',
    rebuild: bool = False,
) -> bool:
    """创建全文索引（幂等）。

    返回 True 表示索引已就绪（已存在或新建成功）；返回 False 表示创建失败（已打印警告）。

    降级逻辑：
    - 带 parser 失败 → 重试无 parser；
    - 无 parser 的首次失败（parser=None）→ 视为可选功能，打印警告并返回 False；
    - 无 parser 的兜底重试失败 → 打印警告并返回 False，不再静默吞掉。
    调用方应收集所有 False 结果并在所有步骤完成后统一抛出，确保 fail-fast。
    """
    if await index_exists(table_name, index_name):
        if rebuild:
            await drop_index_if_exists(table_name, index_name)
        else:
            print_info(f"{table_name}.{index_name}: already exists")
            return True

    column_sql = ", ".join(columns)
    parser_sql = ""
    if parser:
        parser_sql = f" WITH PARSER {parser}"
        if parser_properties:
            parser_sql += f" PARSER_PROPERTIES=({parser_properties})"

    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    f"ADD FULLTEXT INDEX {index_name} ({column_sql})"
                    f"{parser_sql}"
                )
            )
        parser_label = f" with parser {parser}" if parser else ""
        print_success(f"{table_name}.{index_name}: created{parser_label}")
        return True
    except Exception as exc:
        if not parser:
            print_warning(f"{table_name}.{index_name}: fulltext index skipped: {exc}")
            return False
        print_warning(
            f"{table_name}.{index_name}: fulltext index with parser {parser} "
            f"failed, retrying without parser: {exc}"
        )
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(f"ALTER TABLE {table_name} ADD FULLTEXT INDEX {index_name} ({column_sql})")
                )
        except Exception as fallback_exc:
            print_warning(
                f"{table_name}.{index_name}: fulltext index failed without parser too: {fallback_exc}"
            )
            return False
    print_success(f"{table_name}.{index_name}: created without parser")
    return True


async def initialize_oceanbase_vectors(rebuild_fulltext: bool = False) -> None:
    settings = get_settings()
    if settings.effective_database_backend != "oceanbase":
        raise RuntimeError("OceanBase vector initialization requires DATABASE_BACKEND=oceanbase")
    if settings.effective_vector_backend != "oceanbase":
        raise RuntimeError("OceanBase vector initialization requires VECTOR_BACKEND=oceanbase")

    dims = settings.embedding_dimensions or 1024

    print_header("OceanBase vector columns initialization")
    print_info(f"database={settings.oceanbase_database}")
    print_info(f"vector_dims={dims}")
    print_info(f"vector_index_params={build_vector_index_params()}")
    print_info(f"vector_search_ef_search={settings.oceanbase_vector_search_ef_search}")

    await add_column_if_missing(
        "source_chunk",
        "heading_vector",
        f"heading_vector VECTOR({dims}) NULL COMMENT 'heading embedding vector'",
        expected_type=f"vector({dims})",
    )
    await add_column_if_missing(
        "source_chunk",
        "content_vector",
        f"content_vector VECTOR({dims}) NULL COMMENT 'content embedding vector'",
        expected_type=f"vector({dims})",
    )
    await add_column_if_missing(
        "source_event",
        "title_vector",
        f"title_vector VECTOR({dims}) NULL COMMENT 'event title embedding vector'",
        expected_type=f"vector({dims})",
    )
    await add_column_if_missing(
        "source_event",
        "content_vector",
        f"content_vector VECTOR({dims}) NULL COMMENT 'event content embedding vector'",
        expected_type=f"vector({dims})",
    )
    await add_column_if_missing(
        "source_event",
        "entities",
        "entities JSON NULL COMMENT 'search-friendly event entity id list'",
    )
    await add_column_if_missing(
        "entity",
        "vector",
        f"vector VECTOR({dims}) NULL COMMENT 'entity name embedding vector'",
        expected_type=f"vector({dims})",
    )
    await add_column_if_missing(
        "event_entity",
        "vector",
        f"vector VECTOR({dims}) NULL COMMENT 'event-entity relation embedding vector'",
        expected_type=f"vector({dims})",
    )

    print_header("OceanBase vector indexes initialization")
    await create_vector_index_if_missing(
        "source_chunk",
        "heading_vector",
        "idx_obv_source_chunk_heading",
    )
    await create_vector_index_if_missing(
        "source_chunk",
        "content_vector",
        "idx_obv_source_chunk_content",
    )
    await create_vector_index_if_missing(
        "source_event",
        "title_vector",
        "idx_obv_source_event_title",
    )
    await create_vector_index_if_missing(
        "source_event",
        "content_vector",
        "idx_obv_source_event_content",
    )
    await create_vector_index_if_missing(
        "entity",
        "vector",
        "idx_obv_entity_vector",
    )
    await create_vector_index_if_missing(
        "event_entity",
        "vector",
        "idx_obv_event_entity_vector",
    )

    print_header("OceanBase fulltext indexes initialization")
    fulltext_results = []
    fulltext_results.append(
        await create_fulltext_index_if_missing(
            "entity",
            "idx_obft_entity_name",
            ["name", "normalized_name"],
            rebuild=rebuild_fulltext,
        )
    )
    fulltext_results.append(
        await create_fulltext_index_if_missing(
            "source_event",
            "idx_obft_source_event_text",
            ["title", "content"],
            rebuild=rebuild_fulltext,
        )
    )
    fulltext_results.append(
        await create_fulltext_index_if_missing(
            "source_chunk",
            "idx_obft_source_chunk_text",
            ["heading", "content"],
            rebuild=rebuild_fulltext,
        )
    )

    if not all(fulltext_results):
        raise RuntimeError(
            "部分全文索引创建失败（详见上方 [WARN] 输出）——OceanBase 初始化未完全成功"
        )

    print_success("OceanBase vector/fulltext columns and indexes are ready")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize OceanBase vector/fulltext indexes.")
    parser.add_argument(
        "--rebuild-fulltext",
        action="store_true",
        help="Drop and recreate OceanBase fulltext indexes, useful when switching to IK parser.",
    )
    args = parser.parse_args()

    try:
        await initialize_oceanbase_vectors(rebuild_fulltext=args.rebuild_fulltext)
        print_header("Done")
    finally:
        from pipeline.storage import close_storage_facade

        await close_storage_facade()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
