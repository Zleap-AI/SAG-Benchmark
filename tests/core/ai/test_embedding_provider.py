import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from pipeline.core.ai.embedding_provider import (
    EmbeddingClientProvider,
    ResolvedEmbeddingConfig,
)
from pipeline.exceptions import ConfigError


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def make_config(model: str = "embedding-a") -> ResolvedEmbeddingConfig:
    return ResolvedEmbeddingConfig(
        model=model,
        base_url="http://embedding.test/v1",
        api_key="fixture-secret",
        dimensions=1024,
    )


@pytest.mark.asyncio
async def test_concurrent_get_creates_one_shared_client() -> None:
    created: list[FakeEmbeddingClient] = []

    async def resolve(
        _scenario: str, _overrides: Mapping[str, Any] | None
    ) -> ResolvedEmbeddingConfig:
        return make_config()

    def create(_config: ResolvedEmbeddingConfig) -> FakeEmbeddingClient:
        client = FakeEmbeddingClient()
        created.append(client)
        return client

    provider = EmbeddingClientProvider(resolve, create)
    clients = await asyncio.gather(*(provider.get() for _ in range(50)))

    assert len(created) == 1
    assert all(client is created[0] for client in clients)


@pytest.mark.asyncio
async def test_failed_creation_does_not_poison_provider() -> None:
    attempts = 0

    async def resolve(
        _scenario: str, _overrides: Mapping[str, Any] | None
    ) -> ResolvedEmbeddingConfig:
        return make_config()

    def create(_config: ResolvedEmbeddingConfig) -> FakeEmbeddingClient:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("factory failed")
        return FakeEmbeddingClient()

    provider = EmbeddingClientProvider(resolve, create)
    with pytest.raises(RuntimeError, match="factory failed"):
        await provider.get()

    assert isinstance(await provider.get(), FakeEmbeddingClient)
    assert attempts == 2


@pytest.mark.asyncio
async def test_config_change_requires_explicit_reset() -> None:
    model = "embedding-a"

    async def resolve(
        _scenario: str, _overrides: Mapping[str, Any] | None
    ) -> ResolvedEmbeddingConfig:
        return make_config(model)

    provider = EmbeddingClientProvider(resolve, lambda _config: FakeEmbeddingClient())
    await provider.get()
    model = "embedding-b"

    with pytest.raises(ConfigError, match="configuration changed"):
        await provider.get()


@pytest.mark.asyncio
async def test_reset_closes_once_and_allows_recreation() -> None:
    created: list[FakeEmbeddingClient] = []

    async def resolve(
        _scenario: str, _overrides: Mapping[str, Any] | None
    ) -> ResolvedEmbeddingConfig:
        return make_config()

    def create(_config: ResolvedEmbeddingConfig) -> FakeEmbeddingClient:
        client = FakeEmbeddingClient()
        created.append(client)
        return client

    provider = EmbeddingClientProvider(resolve, create)
    first = await provider.get()
    await provider.reset()
    await provider.reset()
    second = await provider.get()

    assert isinstance(first, FakeEmbeddingClient)
    assert first.close_calls == 1
    assert second is not first
    assert len(created) == 2


@pytest.mark.asyncio
async def test_owned_client_does_not_change_shared_state() -> None:
    created: list[FakeEmbeddingClient] = []

    async def resolve(
        _scenario: str, _overrides: Mapping[str, Any] | None
    ) -> ResolvedEmbeddingConfig:
        return make_config()

    def create(_config: ResolvedEmbeddingConfig) -> FakeEmbeddingClient:
        client = FakeEmbeddingClient()
        created.append(client)
        return client

    provider = EmbeddingClientProvider(resolve, create)
    shared = await provider.get()
    owned = await provider.create_owned(overrides={"model": "owned"})

    assert owned is not shared
    assert await provider.get() is shared
    assert len(created) == 2


def test_config_repr_does_not_expose_api_key() -> None:
    config = make_config()
    assert "fixture-secret" not in repr(config)
