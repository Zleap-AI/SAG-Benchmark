"""Regression test: ``chat()`` must tolerate ``response.usage is None``.

Some OpenAI-compatible backends (vLLM and various gateways) do not populate
``usage`` on the completion response. A bare dereference of ``usage`` turns an
otherwise successful call into an ``AttributeError`` and reports it as a failed
request. This guards against that regression.
"""

from types import SimpleNamespace

import pytest

from pipeline.core.ai.llm import OpenAIClient
from pipeline.core.ai.models import LLMMessage, LLMProvider, LLMRole, ModelConfig


class _UsageNoneCompletions:
    async def create(self, **_kwargs):
        message = SimpleNamespace(content="ok", reasoning_content=None, reasoning=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], model="test-model", usage=None)


def _build_client() -> OpenAIClient:
    client = OpenAIClient.__new__(OpenAIClient)
    client.config = ModelConfig(
        provider=LLMProvider.OPENAI,
        model="test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        temperature=0.0,
        max_tokens=1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout=1,
        max_retries=0,
    )
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=_UsageNoneCompletions()))
    return client


@pytest.mark.asyncio
async def test_chat_tolerates_none_usage():
    client = _build_client()

    response = await client.chat([LLMMessage(role=LLMRole.USER, content="test")])

    assert response.content == "ok"
    assert response.model == "test-model"
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0
