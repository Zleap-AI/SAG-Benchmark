from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipeline.modules.search.sag2.rerank import SAG2RerankStage
from pipeline.modules.search.sag2.timing import SAG2TimingService


def _config(
    *,
    max_results: int = 3,
    rerank_top_k: int | None = None,
    rerank_score_threshold: float = 0.0,
    score_threshold: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        sag2_rerank=SimpleNamespace(
            strategy="rerank",
            max_results=max_results,
            rerank_top_k=rerank_top_k or max_results,
            rerank_timeout=1.0,
            rerank_score_threshold=rerank_score_threshold,
            score_threshold=score_threshold,
        )
    )


def _event_details(event_ids: list[str]) -> dict[str, dict[str, str]]:
    return {
        event_id: {
            "event_id": event_id,
            "title": f"title {event_id}",
            "content": f"content {event_id}",
        }
        for event_id in event_ids
    }


def _build_stage(
    *,
    results: list[dict] | None = None,
    error: Exception | None = None,
) -> tuple[SAG2RerankStage, AsyncMock]:
    rerank = AsyncMock(side_effect=error) if error else AsyncMock(return_value=results or [])
    runtime = SimpleNamespace()
    stage = SAG2RerankStage(
        runtime=runtime,
        timing=SAG2TimingService(runtime, ctx_var_name=f"test_rerank_{id(rerank)}"),
        rerank_client=SimpleNamespace(rerank=rerank),
        settings=SimpleNamespace(server_type="REMOTE"),
    )
    return stage, rerank


@pytest.mark.asyncio
async def test_partial_rerank_is_filled_by_stable_embedding_order():
    event_ids = ["e1", "e3", "e2", "e4"]
    event_scores = {"e1": 0.4, "e3": 0.9, "e2": 0.9, "e4": 0.7}
    stage, rerank = _build_stage(
        results=[
            {"id": "e1", "score": 0.95},
            {"id": "e4", "score": 0.94},
        ],
    )

    events, ids, rerank_scores, failed, _ = await stage.rerank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(max_results=3, rerank_top_k=1, score_threshold=0.5),
        _event_details(event_ids),
    )

    assert failed is False
    assert ids == ["e1", "e3", "e2"]
    assert rerank_scores == {"e1": 0.95}
    assert events == [
        {"event_id": "e1", "score": 0.4, "rerank_score": 0.95},
        {"event_id": "e3", "score": 0.9},
        {"event_id": "e2", "score": 0.9},
    ]
    assert rerank.await_args.kwargs["top_n"] == 1


@pytest.mark.asyncio
async def test_embedding_fills_when_all_rerank_results_are_below_threshold():
    event_ids = ["e1", "e2", "e3", "e4"]
    event_scores = {"e1": 0.6, "e2": 0.9, "e3": 0.4, "e4": 0.8}
    stage, _ = _build_stage(
        results=[
            {"id": "e1", "score": 0.2},
            {"id": "e2", "score": 0.7},
        ],
    )

    events, ids, rerank_scores, failed, _ = await stage.rerank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(
            max_results=3,
            rerank_score_threshold=0.8,
            score_threshold=0.5,
        ),
        _event_details(event_ids),
    )

    assert failed is False
    assert ids == ["e2", "e4", "e1"]
    assert rerank_scores == {}
    assert all("rerank_score" not in event for event in events)


@pytest.mark.asyncio
async def test_invalid_duplicate_and_unknown_rerank_items_do_not_pollute_results():
    event_ids = ["e1", "e2", "e3", "e4"]
    event_scores = {"e1": 0.8, "e2": 0.7, "e3": 0.6, "e4": 0.95}
    stage, _ = _build_stage(
        results=[
            {"id": "missing", "score": 0.99},
            {"id": "", "score": 0.98},
            {"id": "e1", "score": "not-a-number"},
            {"id": "e2", "score": "0.91"},
            {"id": "e2", "score": 0.99},
            {"id": "e3", "score": None},
            {"id": "e4", "score": "nan"},
        ],
    )

    events, ids, rerank_scores, failed, _ = await stage.rerank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(max_results=3),
        _event_details(event_ids),
    )

    assert failed is False
    assert ids == ["e2", "e4", "e1"]
    assert rerank_scores == {"e2": 0.91}
    assert [event.get("rerank_score") for event in events] == [0.91, None, None]


@pytest.mark.asyncio
async def test_sufficient_rerank_results_are_capped_without_embedding_fill():
    event_ids = ["e1", "e2", "e3", "e4"]
    event_scores = {"e1": 0.99, "e2": 0.8, "e3": 0.7, "e4": 0.6}
    stage, rerank = _build_stage(
        results=[
            {"id": "e4", "score": 0.95},
            {"id": "e2", "score": 0.9},
            {"id": "e1", "score": 0.85},
        ],
    )

    events, ids, rerank_scores, failed, _ = await stage.rerank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(max_results=2),
        _event_details(event_ids),
    )

    assert failed is False
    assert ids == ["e4", "e2"]
    assert rerank_scores == {"e4": 0.95, "e2": 0.9}
    assert all("rerank_score" in event for event in events)
    assert rerank.await_args.kwargs["top_n"] == 2


@pytest.mark.asyncio
async def test_rank_display_ids_exclude_embedding_fills():
    event_ids = ["e1", "e2", "e3"]
    event_scores = {"e1": 0.5, "e2": 0.9, "e3": 0.8}
    stage, _ = _build_stage(
        results=[{"id": "e1", "score": 0.95}],
    )

    (
        events,
        ids,
        rerank_scores,
        failed,
        strategy,
        _,
        display_ids,
        reason,
        retries,
    ) = await stage.rank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(max_results=3),
        _event_details(event_ids),
    )

    assert ids == ["e1", "e2", "e3"]
    assert [event["event_id"] for event in events] == ids
    assert rerank_scores == {"e1": 0.95}
    assert display_ids == ["e1"]
    assert failed is False
    assert strategy == "rerank"
    assert reason is None
    assert retries == 0


@pytest.mark.asyncio
async def test_rerank_exception_keeps_existing_whole_similarity_fallback():
    event_ids = ["e1", "e2", "e3"]
    event_scores = {"e1": 0.4, "e2": 0.5, "e3": 0.99}
    stage, _ = _build_stage(error=TimeoutError("timed out"))

    events, ids, rerank_scores, failed, _ = await stage.rerank_events(
        "query",
        event_ids,
        event_scores,
        [],
        _config(max_results=2, score_threshold=0.95),
        _event_details(event_ids),
    )

    assert failed is True
    assert ids == ["e1", "e2"]
    assert rerank_scores == {}
    assert events == [
        {"event_id": "e1", "score": 0.4},
        {"event_id": "e2", "score": 0.5},
    ]
