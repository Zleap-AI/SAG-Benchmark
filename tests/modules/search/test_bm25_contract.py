"""Behavior contract for the BM25 chunk search strategy.

These tests intentionally describe the public strategy result rather than the
Elasticsearch implementation hidden behind it.  They protect the migration to
the storage provider boundary from changing ranking, filtering, or result
shape.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_bm25_preserves_query_filters_ranking_and_result_shape():
    from pipeline.modules.search import bm25

    repository = SimpleNamespace(
        search_chunks_by_text=AsyncMock(
            return_value=[
                {
                    "chunk_id": "chunk-mid",
                    "source_id": "source-mid",
                    "source_config_id": "config-a",
                    "heading": "Middle",
                    "content": "middle content",
                    "rank": 2,
                    "_score": 1.5,
                },
                {
                    "chunk_id": "chunk-low",
                    "source_id": "source-low",
                    "source_config_id": "config-a",
                    "heading": "Low",
                    "content": "low content",
                    "rank": 3,
                    "_score": 0.5,
                },
                {
                    "chunk_id": "chunk-high",
                    "source_id": "source-high",
                    "source_config_id": "config-b",
                    "heading": "High",
                    "content": "high content",
                    "rank": 1,
                    "_score": 2.0,
                },
            ]
        )
    )

    searcher = bm25.BM25ChunkSearcher(chunk_text_store=repository)
    result = await searcher.search_for_rerank(
        query="event universe",
        source_config_ids=["config-a", "config-b"],
        config=SimpleNamespace(top_k=2, similarity_threshold=1.0),
    )

    repository.search_chunks_by_text.assert_awaited_once_with(
        query="event universe",
        source_config_ids=["config-a", "config-b"],
        size=2,
    )
    assert result["sections"] == [
        {
            "chunk_id": "chunk-high",
            "source_id": "source-high",
            "source_config_id": "config-b",
            "heading": "High",
            "content": "high content",
            "rank": 1,
            "score": 2.0,
            "weight": 2.0,
        },
        {
            "chunk_id": "chunk-mid",
            "source_id": "source-mid",
            "source_config_id": "config-a",
            "heading": "Middle",
            "content": "middle content",
            "rank": 2,
            "score": 1.5,
            "weight": 1.5,
        },
    ]
    assert set(result["_timings"]) == {"es_search", "total"}


@pytest.mark.asyncio
async def test_bm25_empty_query_does_not_call_storage():
    from pipeline.modules.search import bm25

    repository = SimpleNamespace(search_chunks_by_text=AsyncMock())

    searcher = bm25.BM25ChunkSearcher(chunk_text_store=repository)
    result = await searcher.search_for_rerank(
        query="   ",
        source_config_ids=["config-a"],
    )

    repository.search_chunks_by_text.assert_not_awaited()
    assert result["sections"] == []
    assert result["_timings"]["es_search"] == 0.0
