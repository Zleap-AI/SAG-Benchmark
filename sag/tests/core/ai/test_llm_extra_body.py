"""Tests for OpenAIClient extra_body construction — endpoint-dispatch thinking switch."""

from pipeline.core.ai.models import LLMProvider, ModelConfig


def _make_config(base_url: str, enable_thinking: bool) -> ModelConfig:
    return ModelConfig(
        provider=LLMProvider.OPENAI,
        model="test-model",
        api_key="sk-test",
        base_url=base_url,
        temperature=0.0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        enable_thinking=enable_thinking,
    )


def _build_thinking_extra_body(config: ModelConfig) -> dict:
    # Isolate the pure logic without instantiating AsyncOpenAI (no network/creds).
    from pipeline.core.ai.llm import OpenAIClient

    client = OpenAIClient.__new__(OpenAIClient)
    client.config = config
    return client._build_thinking_extra_body()


def test_302ai_gateway_only_sends_top_level_enable_thinking():
    cfg = _make_config("https://api.302ai.com/v1", enable_thinking=False)
    body = _build_thinking_extra_body(cfg)
    assert body == {"enable_thinking": False}


def test_gpt_302ai_gateway_only_sends_top_level_enable_thinking():
    cfg = _make_config("https://gpt.302.ai/v1", enable_thinking=False)
    body = _build_thinking_extra_body(cfg)
    assert body == {"enable_thinking": False}


def test_vllm_endpoint_sends_chat_template_kwargs():
    cfg = _make_config("http://llm.example.test/v1", enable_thinking=False)
    body = _build_thinking_extra_body(cfg)
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_enable_thinking_true_sends_nothing():
    cfg = _make_config("https://api.302ai.com/v1", enable_thinking=True)
    assert _build_thinking_extra_body(cfg) == {}

    cfg_vllm = _make_config("http://localhost:32768/v1", enable_thinking=True)
    assert _build_thinking_extra_body(cfg_vllm) == {}
