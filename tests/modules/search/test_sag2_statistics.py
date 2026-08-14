import asyncio
from types import SimpleNamespace

import pytest

from pipeline.modules.search.sag2 import (
    SAG2_EVENT_STATS_KEYS,
    SAG2_TIMING_STAGE_ORDER,
)
from pipeline.modules.search.sag2.candidate_scope import SAG2CandidateSubgraph
from pipeline.modules.search.sag2.timing import SAG2TimingService
from pipeline.modules.search.searcher import SAGSearcher


def test_schema_v2_payload_is_additive_and_retry_waste_is_not_double_counted():
    raw = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
    raw.update({"rewrite_query": 4.0, "parallel_recall": 2.0, "finalize": 1.0})
    wasted = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
    wasted["rewrite_query"] = 1.25

    payload = SAG2TimingService.build_timing_payload(
        raw,
        wasted,
        wall_total_observed=7.0,
        step7_timing={
            "with_retry": 3.0,
            "no_retry": 1.75,
            "calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )

    assert payload["schema_version"] == 2
    assert payload["stage_order"] == list(SAG2_TIMING_STAGE_ORDER)
    assert sum(payload["stages_with_retry"].values()) == pytest.approx(
        payload["total_with_retry"], abs=1e-6
    )
    assert sum(payload["stages_no_retry"].values()) == pytest.approx(
        payload["total_no_retry"], abs=1e-6
    )
    assert payload["total_with_retry"] - payload["total_no_retry"] == pytest.approx(
        payload["retry_wasted_total"], abs=1e-6
    )
    assert payload["stages_with_retry"]["rewrite_query"] == 4.0
    assert payload["stages_no_retry"]["rewrite_query"] == 2.75
    assert payload["retry_wasted_by_stage"]["rewrite_query"] == 1.25


def test_scope_modes_share_identical_event_schema_and_zero_disabled_scope_counts():
    common = {
        "query_events": [{"event_id": "q1"}],
        "entity_events": {"e1": ["x"]},
        "expand_event_ids": ["x1"],
        "event_ids": ["q1", "e1", "x1"],
        "score_top_event_ids": ["q1", "e1"],
        "rank_event_ids": ["q1"],
        "seen_event_ids": {"q1", "e1", "x1"},
        "query_entity_ids": ["x"],
        "new_entity_ids": ["y"],
    }
    disabled = SAG2TimingService.build_event_stats(**common, scope=None)
    scope = SAG2CandidateSubgraph(
        event_scores={"q1": 0.9, "e1": 0.8},
        event_to_entities={"q1": ["x"], "e1": ["x", "y"]},
        entity_to_events={"x": ["q1", "e1"], "y": ["e1"]},
    )
    enabled = SAG2TimingService.build_event_stats(**common, scope=scope)

    assert tuple(disabled) == SAG2_EVENT_STATS_KEYS
    assert tuple(enabled) == SAG2_EVENT_STATS_KEYS
    assert disabled["candidate_scope_event_count"] == 0
    assert disabled["candidate_scope_entity_count"] == 0
    assert disabled["candidate_scope_edge_count"] == 0
    assert enabled["candidate_scope_event_count"] == 2
    assert enabled["candidate_scope_entity_count"] == 2
    assert enabled["candidate_scope_edge_count"] == 3


@pytest.mark.asyncio
async def test_request_llm_timing_is_isolated_across_concurrent_tasks():
    timing_service = SAG2TimingService(
        object(), ctx_var_name="test_sag2_concurrent_timing"
    )

    async def request(wasted: float, calls: int) -> tuple[float, int]:
        token = timing_service.start_request()
        try:
            current = timing_service.context_var.get()
            assert current is not None
            current["wasted_retry_time"] = wasted
            current["calls"] = calls
            await asyncio.sleep(0)
            after_yield = timing_service.context_var.get()
            assert after_yield is not None
            return timing_service.retry_wasted_snapshot(), int(after_yield["calls"])
        finally:
            timing_service.context_var.reset(token)

    first, second = await asyncio.gather(request(1.25, 1), request(7.5, 3))

    assert first == (1.25, 1)
    assert second == (7.5, 3)
    assert timing_service.context_var.get() is None


def test_searcher_event_stats_prefers_v2_and_falls_back_to_flat_or_nested_payloads():
    direct = {"query_event_count": 3}
    assert SAGSearcher._get_sag2_event_stats(
        {"event_stats": direct, "stats": {"query_event_count": 99}}
    ) == direct
    assert SAGSearcher._get_sag2_event_stats(
        {"stats": {"query_event_count": 4}}
    ) == {"query_event_count": 4}
    assert SAGSearcher._get_sag2_event_stats(
        {"stats": {"sag2": {"query_event_count": 5}}}
    ) == {"query_event_count": 5}


def test_failed_llm_call_counts_once_without_reusing_success_tokens():
    llm_client = SimpleNamespace(
        _last_call_timing={
            "success_time": 0.0,
            "total_time": 2.5,
            "wasted_retry_time": 2.5,
            "retries": 2,
            "failed": True,
        },
        _last_call_usage=None,
    )
    timing_service = SAG2TimingService(
        SimpleNamespace(llm_client=llm_client),
        ctx_var_name="test_sag2_failed_llm_timing",
    )
    token = timing_service.start_request()
    try:
        timing_service.record_llm_call_metrics()
        timing = timing_service.context_var.get()
        assert timing is not None
        assert timing["calls"] == 1
        assert timing["with_retry"] == 2.5
        assert timing["no_retry"] == 0.0
        assert timing["wasted_retry_time"] == 2.5
        assert timing["prompt_tokens"] == 0
        assert timing["completion_tokens"] == 0
    finally:
        timing_service.context_var.reset(token)
