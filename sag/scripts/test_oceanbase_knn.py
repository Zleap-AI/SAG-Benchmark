#!/usr/bin/env python3
"""Smoke test vector KNN through the unified StorageFacade vector interface."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.core.ai.factory import close_all_clients, get_embedding_client
from pipeline.core.config import get_settings
from pipeline.db.base import close_database
from pipeline.storage import get_storage_facade

TABLE_METHODS = {
    "source_chunk": "search_chunks_by_vector",
    "source_event": "search_events_by_vector",
    "entity": "search_entities_by_vector",
    "event_entity": "search_event_entities_by_vector",
}


def score_from_distance(distance: Any) -> float:
    if distance is None:
        return 0.0
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 1.0
    return 1.0 / (1.0 + value)


def normalize_row(table_name: str, row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("_score", score_from_distance(normalized.get("distance")))

    if table_name == "source_chunk":
        normalized.setdefault("row_id", normalized.get("chunk_id") or normalized.get("id"))
        normalized.setdefault("title", normalized.get("heading"))
        normalized.setdefault("preview", normalized.get("content"))
    elif table_name == "source_event":
        normalized.setdefault("row_id", normalized.get("event_id") or normalized.get("id"))
        normalized.setdefault("title", normalized.get("title"))
        normalized.setdefault("preview", normalized.get("content"))
    elif table_name == "entity":
        normalized.setdefault("row_id", normalized.get("entity_id") or normalized.get("id"))
        normalized.setdefault("title", normalized.get("name"))
        normalized.setdefault("preview", normalized.get("description"))
    elif table_name == "event_entity":
        normalized.setdefault(
            "row_id",
            normalized.get("association_id")
            or normalized.get("event_entity_id")
            or normalized.get("id"),
        )
        title_parts = [
            str(normalized.get("entity_id") or ""),
            str(normalized.get("event_id") or ""),
        ]
        normalized.setdefault("title", " @ ".join(part for part in title_parts if part))
        normalized.setdefault("preview", normalized.get("description"))

    return normalized


def print_rows(
    table_name: str,
    rows: list[dict[str, Any]],
    mode: str,
    fallback_error: str | None = None,
) -> None:
    print(f"\n=== {table_name} ===")
    print(f"mode={mode}")
    if fallback_error:
        print(f"fallback_error={fallback_error}")
    if not rows:
        print("No vector rows found.")
        return

    for idx, row in enumerate(rows, 1):
        distance = row.get("distance")
        score = row.get("_score", score_from_distance(distance))
        title = (row.get("title") or "").replace("\n", " ")[:120]
        preview = (row.get("preview") or "").replace("\n", " ")[:180]
        print(f"{idx}. id={row.get('row_id')}")
        print(f"   source_config_id={row.get('source_config_id')}")
        print(f"   distance={distance} score={score:.6f}")
        if title:
            print(f"   title={title}")
        if preview:
            print(f"   preview={preview}")


async def search_table(
    table_name: str,
    query_vector: list[float],
    top_k: int,
    source_config_id: str | None,
    exact: bool,
) -> tuple[list[dict[str, Any]], str, str | None]:
    vector_store = get_storage_facade().vector
    method = getattr(vector_store, TABLE_METHODS[table_name])
    kwargs: dict[str, Any] = {
        "query_vector": query_vector,
        "k": top_k,
        "exact": exact,
    }
    if source_config_id:
        kwargs["source_config_id"] = source_config_id
    try:
        rows = await method(**kwargs)
        mode = "exact" if exact else "ann"
        fallback_error = None
    except Exception as exc:
        if exact:
            raise
        fallback_error = str(exc)
        kwargs["exact"] = True
        rows = await method(**kwargs)
        mode = "exact_fallback"
    return [normalize_row(table_name, row) for row in rows], mode, fallback_error


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test 4-table KNN via StorageFacade.vector")
    parser.add_argument("--query", default="Who is Lionel Messi?", help="Query text to embed")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results per table")
    parser.add_argument("--source-config-id", default=None, help="Optional source_config_id filter")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Pass exact=True to providers that support exact vector ordering",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-readable text",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    settings = get_settings()
    if settings.effective_vector_backend != "oceanbase":
        print(
            "Current VECTOR_BACKEND is not oceanbase. "
            f"effective_vector_backend={settings.effective_vector_backend}",
            file=sys.stderr,
        )
        return 2

    try:
        embedding_client = await get_embedding_client(scenario="general")
        query_vector = await embedding_client.generate(args.query)

        output: dict[str, Any] = {
            "query": args.query,
            "embedding_dimensions": len(query_vector),
            "vector_backend": get_storage_facade().vector.backend_name,
            "source_config_id": args.source_config_id,
            "exact": args.exact,
            "tables": {},
        }

        for table_name in TABLE_METHODS:
            rows, mode, fallback_error = await search_table(
                table_name=table_name,
                query_vector=query_vector,
                top_k=args.top_k,
                source_config_id=args.source_config_id,
                exact=args.exact,
            )
            output["tables"][table_name] = {
                "mode": mode,
                "fallback_error": fallback_error,
                "rows": rows,
            }

        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"Query: {args.query}")
            print(f"Embedding dimensions: {len(query_vector)}")
            print(f"Vector backend: {output['vector_backend']}")
            if args.source_config_id:
                print(f"source_config_id: {args.source_config_id}")
            for table_name, payload in output["tables"].items():
                print_rows(
                    table_name,
                    payload["rows"],
                    mode=payload["mode"],
                    fallback_error=payload["fallback_error"],
                )

        return 0
    finally:
        from pipeline.storage import close_storage_facade

        await close_all_clients()
        await close_storage_facade()
        await close_database()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
