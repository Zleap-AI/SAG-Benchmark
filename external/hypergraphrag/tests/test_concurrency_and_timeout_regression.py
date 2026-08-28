"""Regression tests for two HyperGraphRAG runtime bugs that fail silently.

Both bugs degrade throughput without raising anything, so they need automated
guards rather than a human watching a progress bar:

1. ``limit_async_func_call`` leaked one concurrency slot on every exception
   (upstream bug — no ``try/finally`` around the wrapped call). A long index run
   that hit ``APITimeoutError`` repeatedly would silently degrade effective LLM
   concurrency from ``max_size`` down to 1. Fixed by a local patch that releases
   the slot in ``finally``.

2. ``hypergraphrag_config.llm_model_func`` used ``system_prompt is None`` as a
   proxy for "index path" to pick the LLM timeout, so the QA-stage keyword
   extraction (also ``system_prompt=None``) wrongly inherited the 600s index
   timeout. Fixed by ``set_index_mode()``: the process now picks its timeout
   bucket explicitly, not from the prompt shape.

These tests use an in-process fake HTTP transport — no real LLM/embedding
endpoint is contacted.
"""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hypergraphrag_config as cfg
from hypergraphrag.utils import limit_async_func_call


def _asyncio_run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. limit_async_func_call must release its slot on exception
# ---------------------------------------------------------------------------

def test_limit_async_func_call_releases_slot_after_exception():
    """Peak concurrency stays == max_size even after many raised calls.

    Before the local patch, each raised call skipped ``__current_size -= 1`` and
    permanently leaked a slot: enough failures would drop the ceiling to 1 (or
    hang forever once all slots leaked).
    """
    max_size = 4
    current = 0
    peak = 0

    @limit_async_func_call(max_size)
    async def boom():
        # Simulate the slow erroring call: holds a slot, then fails.
        await asyncio.sleep(0.01)
        raise RuntimeError("simulated APITimeoutError")

    @limit_async_func_call(max_size)
    async def ok():
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        try:
            await asyncio.sleep(0.01)
        finally:
            current -= 1
        return "ok"

    async def scenario():
        # 20 raised calls: before the fix these would consume all max_size slots.
        for _ in range(20):
            with pytest.raises(RuntimeError):
                await boom()
        # After the storm the slot counter must be back to 0, so a fresh burst
        # can once again reach max_size concurrency (not hang, not cap at 1).
        results = await asyncio.gather(*[ok() for _ in range(20)])
        assert results == ["ok"] * 20

    _asyncio_run(scenario())
    assert peak == max_size


def test_limit_async_func_call_releases_slot_after_cancellation():
    """Cancellation must also release the slot (finally covers it)."""

    @limit_async_func_call(1)
    async def controlled(delay):
        await asyncio.sleep(delay)
        return "done"

    async def scenario():
        # Acquire the single slot with a long sleep, then cancel it.
        task = asyncio.ensure_future(controlled(60))
        await asyncio.sleep(0.01)  # let it acquire the slot
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Slot released -> an immediate call on the SAME wrapper must not hang.
        result = await asyncio.wait_for(controlled(0), timeout=1.0)
        assert result == "done"

    _asyncio_run(scenario())


# ---------------------------------------------------------------------------
# 2. llm_model_func timeout bucket follows set_index_mode, not prompt shape
# ---------------------------------------------------------------------------

class _FakeTransport(httpx.AsyncBaseTransport):
    """Records the timeout that actually reached the HTTP layer."""

    def __init__(self):
        self.timeouts = []

    async def handle_async_request(self, request):
        self.timeouts.append(request.extensions.get("timeout"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
            request=request,
        )


@pytest.fixture()
def fake_llm_client(monkeypatch):
    """Swap the module singleton for a client backed by _FakeTransport."""
    transport = _FakeTransport()
    client = AsyncOpenAI(
        base_url="http://fake/v1", api_key="1",
        http_client=httpx.AsyncClient(transport=transport),
    )
    monkeypatch.setattr(cfg, "_llm_client", client)
    return transport


def test_timeout_qa_by_default_and_index_after_set_index_mode(fake_llm_client):
    """QA (default) uses 180s; index mode uses 600s; connect stays 5s.

    The keyword-extraction call passes no system_prompt, so a prompt-shape
    heuristic would wrongly route it to the 600s index bucket. The explicit
    ``_index_mode`` flag keeps it at 180s.
    """

    async def scenario():
        # QA stage — no system_prompt (keyword extraction). Must be 180s, not 600s.
        await cfg.llm_model_func("kw prompt")
        await cfg.llm_model_func("answer", system_prompt="You are helpful")

        # Switch to index mode (what Step_1 does at entry).
        cfg.set_index_mode()
        # Index stage — hint prompt and summary (both system_prompt variants).
        await cfg.llm_model_func("extract prompt")
        await cfg.llm_model_func("summary", system_prompt="Summarize")

    _asyncio_run(scenario())

    qa_kw, qa_ans, idx_hint, idx_sum = fake_llm_client.timeouts

    # request.extensions["timeout"] reaches httpx as a dict, not a Timeout object.
    assert (qa_kw["read"], qa_kw["connect"]) == (180.0, 5.0)
    assert (qa_ans["read"], qa_ans["connect"]) == (180.0, 5.0)
    assert (idx_hint["read"], idx_hint["connect"]) == (600.0, 5.0)
    assert (idx_sum["read"], idx_sum["connect"]) == (600.0, 5.0)


def test_timeout_explicit_call_override_wins(fake_llm_client):
    """An explicit per-call timeout overrides the mode default."""

    async def scenario():
        await cfg.llm_model_func("hi", timeout=42)

    _asyncio_run(scenario())

    (t,) = fake_llm_client.timeouts
    assert t["read"] == 42.0
    assert t["connect"] == 5.0


def test_timeout_explicit_httpx_timeout_passthrough(fake_llm_client):
    """A prebuilt httpx.Timeout is used as-is (no double-wrapping AssertionError)."""

    async def scenario():
        await cfg.llm_model_func("hi", timeout=httpx.Timeout(90.0, connect=1.0))

    _asyncio_run(scenario())

    (t,) = fake_llm_client.timeouts
    assert t["read"] == 90.0
    assert t["connect"] == 1.0
