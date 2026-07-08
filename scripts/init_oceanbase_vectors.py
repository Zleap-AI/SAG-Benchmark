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
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = :database_name
                  AND table_name = :table_name
                  AND column_name = :column_name
                """
            ),
            {
                "database_name": database_name,
                "table_name": table_name,
                "column_name": column_name,
            },
        )
        return int(result.scalar_one()) > 0


async def add_column_if_missing(table_name: str, column_name: str, column_sql: str) -> None:
    if await column_exists(table_name, column_name):
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
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.statistics
                WHERE table_schema = :database_name
                  AND table_name = :table_name
                  AND index_name = :index_name
                """
            ),
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
            text(
                f"CREATE VECTOR INDEX {index_name} "
                f"ON {table_name}({column_name}) "
                f"WITH ({params})"
            )
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
) -> None:
    if await index_exists(table_name, index_name):
        if rebuild:
            await drop_index_if_exists(table_name, index_name)
        else:
            print_info(f"{table_name}.{index_name}: already exists")
            return

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
        return
    except Exception as exc:
        if not parser:
            print_warning(f"{table_name}.{index_name}: fulltext index skipped: {exc}")
            return
        print_warning(
            f"{table_name}.{index_name}: fulltext index with parser {parser} "
            f"failed, retrying without parser: {exc}"
        )
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD FULLTEXT INDEX {index_name} ({column_sql})"
                    )
                )
        except Exception as fallback_exc:
            print_warning(
                f"{table_name}.{index_name}: fulltext index skipped: {fallback_exc}"
            )
            return
    print_success(f"{table_name}.{index_name}: created without parser")


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
    )
    await add_column_if_missing(
        "source_chunk",
        "content_vector",
        f"content_vector VECTOR({dims}) NULL COMMENT 'content embedding vector'",
    )
    await add_column_if_missing(
        "source_event",
        "title_vector",
        f"title_vector VECTOR({dims}) NULL COMMENT 'event title embedding vector'",
    )
    await add_column_if_missing(
        "source_event",
        "content_vector",
        f"content_vector VECTOR({dims}) NULL COMMENT 'event content embedding vector'",
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
    )
    await add_column_if_missing(
        "event_entity",
        "vector",
        f"vector VECTOR({dims}) NULL COMMENT 'event-entity relation embedding vector'",
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
    await create_fulltext_index_if_missing(
        "entity",
        "idx_obft_entity_name",
        ["name", "normalized_name"],
        rebuild=rebuild_fulltext,
    )
    await create_fulltext_index_if_missing(
        "source_event",
        "idx_obft_source_event_text",
        ["title", "content"],
        rebuild=rebuild_fulltext,
    )
    await create_fulltext_index_if_missing(
        "source_chunk",
        "idx_obft_source_chunk_text",
        ["heading", "content"],
        rebuild=rebuild_fulltext,
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
