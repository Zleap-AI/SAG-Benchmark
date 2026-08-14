"""Ownership and shutdown contracts for the SAG2 runtime chain."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipeline.engine.core import pipelineEngine
from pipeline.modules.search.sag2 import SAG2Searcher
from pipeline.modules.search.searcher import SAGSearcher


@pytest.mark.asyncio
async def test_sag2_searcher_closes_its_runtime():
    runtime = SimpleNamespace(aclose=AsyncMock())
    searcher = SAG2Searcher(runtime=runtime)

    await searcher.aclose()

    runtime.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_sag_searcher_closes_only_created_sag2_searcher():
    searcher = SAGSearcher.__new__(SAGSearcher)
    searcher._sag2_searcher = None

    await searcher.aclose()

    owned_sag2 = SimpleNamespace(aclose=AsyncMock())
    searcher._sag2_searcher = owned_sag2
    await searcher.aclose()

    owned_sag2.aclose.assert_awaited_once_with()
    assert searcher._sag2_searcher is None


@pytest.mark.asyncio
async def test_pipeline_engine_closes_its_searcher():
    engine = pipelineEngine.__new__(pipelineEngine)
    engine._searcher = SimpleNamespace(aclose=AsyncMock())

    await engine.aclose()

    engine._searcher.aclose.assert_awaited_once_with()
