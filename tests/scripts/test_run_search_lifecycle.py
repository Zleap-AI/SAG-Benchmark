"""Lifecycle coverage for the search benchmark engine pool."""

from unittest.mock import AsyncMock

import pytest

import pipeline
from scripts.run_search_benchmark import run_batch_search


@pytest.mark.asyncio
async def test_run_batch_search_closes_every_engine(monkeypatch):
    engines = []

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            self.aclose = AsyncMock()
            engines.append(self)

    monkeypatch.setattr(pipeline, "PipelineEngine", _FakeEngine)

    result = await run_batch_search(
        questions=[],
        source_config_id="source-1",
        strategy="sag2",
        mode="section",
        top_k=5,
        max_concurrency=3,
        bench_size=1,
        gold_docs_for_recall=[],
        mlflow_tracker=None,
        bench_logger=None,
    )

    assert result == []
    assert len(engines) == 3
    for engine in engines:
        engine.aclose.assert_awaited_once_with()
