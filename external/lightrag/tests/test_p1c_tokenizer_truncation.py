"""LightRAG embedding-token truncation regression tests."""

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_short_text_is_verified_with_embedding_tokenizer(monkeypatch):
    import lightrag_config as config

    counter = AsyncMock(return_value=2)
    monkeypatch.setattr(config, "_embedding_token_count", counter)

    assert await config._truncate_text_by_tokens("中文", 512) == "中文"
    counter.assert_awaited_once_with("中文")


@pytest.mark.asyncio
async def test_binary_search_returns_longest_fitting_prefix(monkeypatch):
    import lightrag_config as config

    async def token_count(text):
        return len(text)

    monkeypatch.setattr(config, "_embedding_token_count", token_count)
    assert await config._truncate_text_by_tokens("abcdefghij", 5) == "abcde"


@pytest.mark.asyncio
async def test_tokenizer_failure_uses_conservative_fallback(monkeypatch):
    import lightrag_config as config

    counter = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr(config, "_embedding_token_count", counter)

    text = "x" * 5000
    assert await config._truncate_text_by_tokens(text, 512) == text[:1536]


@pytest.mark.asyncio
async def test_batch_preserves_order(monkeypatch):
    import lightrag_config as config

    async def truncate(text, max_tokens):
        return f"{text}:{max_tokens}"

    monkeypatch.setattr(config, "_truncate_text_by_tokens", truncate)
    result = await config._truncate_embeddings_batch(["a", "bbb", "cc"], 12)
    assert result == ["a:12", "bbb:12", "cc:12"]


def test_tokenizer_url_uses_server_root(monkeypatch):
    import lightrag_config as config

    monkeypatch.setenv("EMBED_TOKENIZER_URL", "")
    monkeypatch.setattr(config, "EMBEDDING_BASE_URL", "http://embed:8000/v1")
    assert config._tokenizer_base_url() == "http://embed:8000"


def test_explicit_tokenizer_url_also_strips_v1(monkeypatch):
    import lightrag_config as config

    monkeypatch.setenv("EMBED_TOKENIZER_URL", "http://tokenizer:9000/v1/")
    assert config._tokenizer_base_url() == "http://tokenizer:9000"
