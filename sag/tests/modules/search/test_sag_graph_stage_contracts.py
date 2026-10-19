from __future__ import annotations

from typing import Any

import pytest

from pipeline.modules.search.config import SAGConfig
from pipeline.modules.search.sag_graph import (
    SAGGraphExpandStage,
    SAGGraphRecallResult,
    SAGGraphRecallStage,
    SAGGraphRequest,
    SAGGraphRerankStage,
    SAGGraphSearchState,
)
from pipeline.modules.search.sag_graph.evidence import EvidenceTracker
from pipeline.modules.search.sag_graph.routes import SAGGraphRouteTracker
from pipeline.modules.search.sag_graph.timing import SAGGraphTimingService


class _ExpandRuntime:
    async def search_event_entities_by_vector(self, **kwargs: Any):
        event_id = kwargs["event_ids"][0]
        return [{"event_id": event_id, "entity_id": "entity-2", "score": 0.9}]

    async def retrieve_entity_event_pairs(self, *_: Any, **__: Any):
        return {"event-2": ["entity-2"]}, ["event-2"]

    async def coarse_rank_events(self, **_: Any):
        return [{"event_id": "event-2", "score": 0.8}]


class _Processor:
    async def generate_embedding(self, _: str):
        return [0.25]


class _RecallRuntime:
    async def get_processor(self):
        return _Processor()

    async def search_events_by_vector(self, **_: Any):
        return [{"event_id": "event-1", "entity_ids": ["entity-1"], "score": 0.9}]

    async def search_entities_by_text(self, **_: Any):
        return [{"entity_id": "entity-1"}]

    async def retrieve_entity_event_pairs(self, *_: Any, **__: Any):
        return {"event-2": ["entity-1"]}, ["event-2"]

    async def coarse_rank_events(self, **_: Any):
        return [{"event_id": "event-2", "score": 0.8}]


class _RerankRuntime:
    async def get_events_by_ids(self, event_ids: list[str], **_: Any):
        return [
            {"event_id": event_id, "title": event_id, "content": event_id} for event_id in event_ids
        ]

    async def search_events_by_text(self, **_: Any):
        return [
            {"event_id": "event-2", "_score": 2.0},
            {"event_id": "event-1", "_score": 1.0},
        ]


def _recall_result() -> SAGGraphRecallResult:
    return SAGGraphRecallResult(
        rewritten_query=None,
        rewritten_entities=[],
        effective_query="question",
        query_vector=[0.1],
        scope=None,
        query_events=[{"event_id": "event-1", "score": 0.9}],
        entity_events={},
        entity_event_docs=[],
        query_event_ids=["event-1"],
        entity_event_ids=[],
        query_entity_ids=[],
        event_ids=["event-1"],
        event_scores={"event-1": 0.9},
        initial_route_stats={"seed_event_key_paths": 0},
    )


@pytest.mark.asyncio
async def test_expand_stage_returns_route_stats_for_orchestrator_merge():
    runtime = _ExpandRuntime()
    route_index = SAGGraphRouteTracker.new_route_index("question")
    result = await SAGGraphExpandStage(
        runtime=runtime,
        timing=SAGGraphTimingService(runtime, ctx_var_name="test_sag_expand_timing"),
    ).run(
        request=SAGGraphRequest(
            query="question",
            source_config_ids=["source-1"],
            config=SAGConfig(sag_expand={"enabled": True, "max_hops": 1}),
            gold_evidences=[],
        ),
        recall=_recall_result(),
        state=SAGGraphSearchState(),
        route_index=route_index,
        evidence=EvidenceTracker(set()),
        timings={},
        retry_wasted_by_stage={},
    )

    assert result.route_stats == {
        "seed_event_key_paths": 1,
        "total_edges": 1,
    }
    assert result.event_ids == ["event-1", "event-2"]


@pytest.mark.asyncio
async def test_expand_stage_disabled_has_zero_route_delta():
    runtime = _ExpandRuntime()
    result = await SAGGraphExpandStage(
        runtime=runtime,
        timing=SAGGraphTimingService(runtime, ctx_var_name="test_sag_expand_disabled_timing"),
    ).run(
        request=SAGGraphRequest(
            query="question",
            source_config_ids=["source-1"],
            config=SAGConfig(sag_expand={"enabled": False}),
            gold_evidences=[],
        ),
        recall=_recall_result(),
        state=SAGGraphSearchState(),
        route_index={},
        evidence=EvidenceTracker(set()),
        timings={},
        retry_wasted_by_stage={},
    )

    assert result.route_stats == {"seed_event_key_paths": 0}
    assert result.event_ids == ["event-1"]


@pytest.mark.asyncio
async def test_recall_stage_runs_without_sag_searcher_owner():
    runtime = _RecallRuntime()
    route_index = SAGGraphRouteTracker.new_route_index("question")
    result = await SAGGraphRecallStage(
        runtime=runtime,
        timing=SAGGraphTimingService(runtime, ctx_var_name="test_sag_recall_timing"),
    ).run(
        request=SAGGraphRequest(
            query="question",
            source_config_ids=["source-1"],
            config=SAGConfig(
                sag_rewrite_query_enabled=False,
                sag_enable_entity_extraction=False,
            ),
            gold_evidences=[],
        ),
        state=SAGGraphSearchState(),
        route_index=route_index,
        evidence=EvidenceTracker(set()),
        timings={},
        retry_wasted_by_stage={},
    )

    assert result.event_ids == ["event-1", "event-2"]
    assert result.event_scores == {"event-1": 0.9, "event-2": 0.8}
    assert result.initial_route_stats["query_event_paths"] == 1
    assert result.initial_route_stats["entity_event_paths"] == 1


@pytest.mark.asyncio
async def test_rerank_stage_runs_without_sag_searcher_owner():
    runtime = _RerankRuntime()
    result = await SAGGraphRerankStage(
        runtime=runtime,
        timing=SAGGraphTimingService(runtime, ctx_var_name="test_sag_rerank_timing"),
    ).run(
        request=SAGGraphRequest(
            query="question",
            source_config_ids=["source-1"],
            config=SAGConfig(
                sag_rerank={"strategy": "rrf", "max_results": 2},
            ),
            gold_evidences=[],
        ),
        event_ids=["event-1", "event-2"],
        event_scores={"event-1": 0.9, "event-2": 0.8},
        event_details=None,
        evidence=EvidenceTracker(set()),
        timings={},
        retry_wasted_by_stage={},
    )

    assert result.rank_method == "rrf"
    assert result.rank_failed is False
    assert result.rank_event_ids == ["event-1", "event-2"]
    assert set(result.rank_scores) == {"event-1", "event-2"}
