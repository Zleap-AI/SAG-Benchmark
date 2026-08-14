import asyncio

import pytest

from pipeline.core.ai.llm import OpenAIClient
from pipeline.utils.llm_tracking import (
    LLMTokenTracker,
    enable_llm_tracking,
    llm_tracking_scope,
    llm_tracking_stage,
    record_llm_usage,
)

USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_scope_records_only_inside_active_context() -> None:
    tracker = LLMTokenTracker()
    await record_llm_usage(USAGE)
    with llm_tracking_scope(tracker), llm_tracking_stage("extract"):
        await record_llm_usage(USAGE)
    await record_llm_usage(USAGE)

    summary = tracker.get_summary()
    assert summary["total_calls"] == 1
    assert summary["total_tokens"] == 5
    assert summary["stages"]["EXTRACT"]["total"] == 5


@pytest.mark.asyncio
async def test_concurrent_contexts_do_not_cross_contaminate() -> None:
    first = LLMTokenTracker()
    second = LLMTokenTracker()

    async def run(tracker: LLMTokenTracker, stage: str, calls: int) -> None:
        with llm_tracking_scope(tracker), llm_tracking_stage(stage):
            await asyncio.gather(*(record_llm_usage(USAGE) for _ in range(calls)))

    await asyncio.gather(run(first, "load", 2), run(second, "qa", 3))

    assert first.get_summary()["stages"]["LOAD"]["calls"] == 2
    assert "QA" not in first.get_summary()["stages"]
    assert second.get_summary()["stages"]["QA"]["calls"] == 3
    assert "LOAD" not in second.get_summary()["stages"]


@pytest.mark.asyncio
async def test_nested_stage_and_exception_restore_context() -> None:
    tracker = LLMTokenTracker()

    with llm_tracking_scope(tracker), llm_tracking_stage("load"):
        await record_llm_usage(USAGE)
        with pytest.raises(RuntimeError):
            with llm_tracking_stage("extract"):
                await record_llm_usage(USAGE)
                raise RuntimeError("stop")
        await record_llm_usage(USAGE)

    summary = tracker.get_summary()
    assert summary["stages"]["LOAD"]["calls"] == 2
    assert summary["stages"]["EXTRACT"]["calls"] == 1


@pytest.mark.asyncio
async def test_compatibility_api_does_not_mutate_openai_method() -> None:
    original = OpenAIClient.chat
    tracker = LLMTokenTracker()
    restore = enable_llm_tracking(tracker)
    try:
        with llm_tracking_stage("search"):
            await record_llm_usage(USAGE)
    finally:
        restore()
        restore()

    assert OpenAIClient.chat is original
    assert tracker.get_summary()["stages"]["SEARCH"]["calls"] == 1
