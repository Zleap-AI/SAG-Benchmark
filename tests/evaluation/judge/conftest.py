"""Fixtures for Judge unit tests."""

import json
from pathlib import Path

import pytest

from pipeline.core.ai.models import LLMMessage, LLMResponse, LLMUsage

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str):
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def gen_samples():
    return _load_fixture("generation_samples.json")


@pytest.fixture(scope="session")
def ret_samples():
    return _load_fixture("retrieval_samples.json")


@pytest.fixture(scope="session")
def fake_responses():
    return _load_fixture("fake_llm_responses.json")


class FakeLLM:
    """Fake LLM client for deterministic unit testing.

    Returns predetermined responses from fake_llm_responses.json.
    Call .set_response() to override per-test.
    """

    def __init__(self, responses: dict | None = None):
        self._responses = responses or _load_fixture("fake_llm_responses.json")
        self._override: dict[str, LLMResponse] = {}
        self.call_count = 0
        self.call_history: list[dict] = []

        class _FakeConfig:
            model = "fake-judge"
            api_key = "sk-fake"
            enable_thinking = False

        self.config = _FakeConfig()

    def set_response(self, key: str, content: str):
        self._override[key] = LLMResponse(
            content=content,
            model="fake-judge",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def set_responses(self, responses: list[str]):
        """Set a queue of responses for sequential calls."""
        self._response_queue = responses
        self._queue_idx = 0

    async def chat(
        self, messages: list[LLMMessage], temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        self.call_count += 1
        self.call_history.append({
            "messages": [{"role": m.role.value, "content": m.content[:200]} for m in messages],
            "temperature": temperature,
        })

        # Queue mode
        if hasattr(self, "_response_queue") and self._queue_idx < len(self._response_queue):
            content = self._response_queue[self._queue_idx]
            self._queue_idx += 1
            return LLMResponse(
                content=content,
                model="fake-judge",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        # Override mode
        for _key, resp in self._override.items():
            return resp

        # Default: return empty JSON object
        return LLMResponse(
            content="{}",
            model="fake-judge",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def close(self):
        pass


def make_fake_llm():
    """Create a FakeLLM preloaded with standard fake responses."""
    responses = _load_fixture("fake_llm_responses.json")
    llm = FakeLLM(responses)

    # Pre-set responses for common calls
    llm.set_response(
        "statement_gen",
        responses["statement_generation"]["llm_response"]["content"],
    )
    llm.set_response(
        "correctness",
        responses["correctness_classification"]["llm_response"]["content"],
    )
    llm.set_response(
        "fact_extraction",
        responses["fact_extraction"]["llm_response"]["content"],
    )
    llm.set_response(
        "fact_coverage",
        responses["fact_coverage"]["llm_response"]["content"],
    )
    return llm
