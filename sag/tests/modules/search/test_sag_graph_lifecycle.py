"""Ownership and shutdown contracts for the SAG runtime chain."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipeline.engine.core import PipelineEngine
from pipeline.modules.search.sag_graph import SAGGraphSearcher
from pipeline.modules.search.searcher import SAGSearcher


@pytest.mark.asyncio
async def test_sag_searcher_closes_its_runtime():
    runtime = SimpleNamespace(aclose=AsyncMock())
    searcher = SAGGraphSearcher(runtime=runtime)

    await searcher.aclose()

    runtime.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_sag_searcher_closes_only_created_sag_searcher():
    searcher = SAGSearcher.__new__(SAGSearcher)
    searcher._sag_searcher = None

    await searcher.aclose()

    owned_sag = SimpleNamespace(aclose=AsyncMock())
    searcher._sag_searcher = owned_sag
    await searcher.aclose()

    owned_sag.aclose.assert_awaited_once_with()
    assert searcher._sag_searcher is None


@pytest.mark.asyncio
async def test_pipeline_engine_closes_its_searcher():
    engine = PipelineEngine.__new__(PipelineEngine)
    engine._searcher = SimpleNamespace(aclose=AsyncMock())

    await engine.aclose()

    engine._searcher.aclose.assert_awaited_once_with()
