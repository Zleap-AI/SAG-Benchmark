from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from pipeline.core.ai.base import LLMRetryClient
from pipeline.exceptions import LLMError


class _Parsed(BaseModel):
    value: str


class _SequenceClient:
    def __init__(self, outcomes, *, max_retries: int):
        self.config = SimpleNamespace(max_retries=max_retries)
        self.outcomes = iter(outcomes)
        self.calls = 0
        self._last_call_usage = None

    async def chat_parsed(self, _messages, _response_model, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        self._last_call_usage = outcome[1]
        return outcome[0]


@pytest.mark.asyncio
async def test_success_then_nonretry_failure_clears_previous_timing_and_usage():
    success_usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    wrapped = _SequenceClient(
        [(_Parsed(value="ok"), success_usage), ValueError("invalid request")],
        max_retries=0,
    )
    client = LLMRetryClient(wrapped)

    assert await client.chat_parsed([], _Parsed) == _Parsed(value="ok")
    previous_timing = dict(client._last_call_timing)
    assert client._last_call_usage is success_usage

    with pytest.raises(ValueError, match="invalid request"):
        await client.chat_parsed([], _Parsed)

    assert client._last_call_usage is None
    assert client._last_call_timing != previous_timing
    assert client._last_call_timing["failed"] is True
    assert client._last_call_timing["success_time"] == 0.0
    assert client._last_call_timing["retries"] == 0
    assert client._last_call_timing["total_time"] >= client._last_call_timing[
        "wasted_retry_time"
    ]


@pytest.mark.asyncio
async def test_exhausted_retries_record_failure_timing_conservation():
    wrapped = _SequenceClient(
        [LLMError("one"), LLMError("two"), LLMError("three")],
        max_retries=2,
    )
    client = LLMRetryClient(wrapped)

    with pytest.raises(LLMError, match="已重试2次"):
        await client.chat_parsed([], _Parsed)

    timing = client._last_call_timing
    assert wrapped.calls == 3
    assert timing["failed"] is True
    assert timing["retries"] == 2
    assert timing["success_time"] == 0.0
    assert timing["total_time"] >= timing["wasted_retry_time"]
    assert timing["total_time"] == pytest.approx(timing["wasted_retry_time"], abs=0.01)
    assert client._last_call_usage is None


@pytest.mark.asyncio
async def test_nonretry_failure_records_zero_retries_and_preserves_exception():
    wrapped = _SequenceClient([ValueError("bad schema")], max_retries=3)
    client = LLMRetryClient(wrapped)

    with pytest.raises(ValueError, match="bad schema"):
        await client.chat_parsed([], _Parsed)

    assert wrapped.calls == 1
    assert client._last_call_timing["failed"] is True
    assert client._last_call_timing["retries"] == 0
    assert client._last_call_timing["success_time"] == 0.0
    assert client._last_call_usage is None
