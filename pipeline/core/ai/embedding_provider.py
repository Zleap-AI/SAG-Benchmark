"""Concurrency-safe ownership and lifecycle for embedding clients."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from pipeline.exceptions import ConfigError


class ManagedEmbeddingClient(Protocol):
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedEmbeddingConfig:
    """Validated immutable configuration used to create an embedding client."""

    model: str
    base_url: str | None
    api_key: str = field(repr=False)
    dimensions: int | None = None
    timeout: float = 60.0
    max_retries: int = 3

    def fingerprint(self) -> str:
        """Return a non-reversible identity used only for equality checks."""
        fields = (
            self.model,
            self.base_url or "",
            hashlib.sha256(self.api_key.encode("utf-8")).hexdigest(),
            str(self.dimensions),
            str(self.timeout),
            str(self.max_retries),
        )
        return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


ConfigResolver = Callable[
    [str, Mapping[str, Any] | None],
    Awaitable[ResolvedEmbeddingConfig],
]
ClientFactory = Callable[[ResolvedEmbeddingConfig], ManagedEmbeddingClient]


class EmbeddingClientProvider:
    """Own exactly one shared client and serialize lifecycle transitions."""

    def __init__(
        self,
        config_resolver: ConfigResolver,
        client_factory: ClientFactory,
    ) -> None:
        self._config_resolver = config_resolver
        self._client_factory = client_factory
        self._lock = asyncio.Lock()
        self._client: ManagedEmbeddingClient | None = None
        self._config_fingerprint: str | None = None

    async def get(self, scenario: str = "general") -> ManagedEmbeddingClient:
        config = await self._config_resolver(scenario, None)
        fingerprint = config.fingerprint()

        async with self._lock:
            if self._client is None:
                client = self._client_factory(config)
                self._client = client
                self._config_fingerprint = fingerprint
            elif fingerprint != self._config_fingerprint:
                raise ConfigError(
                    "Embedding configuration changed while the shared client is active. "
                    "Call 'await reset_embedding_client()' only after in-flight work finishes."
                )
            return self._client

    async def create_owned(
        self,
        scenario: str = "general",
        overrides: Mapping[str, Any] | None = None,
    ) -> ManagedEmbeddingClient:
        """Create a caller-owned client without changing shared provider state."""
        config = await self._config_resolver(scenario, overrides)
        return self._client_factory(config)

    async def reset(self) -> None:
        """Detach and close the shared client. Safe to call repeatedly."""
        async with self._lock:
            client = self._client
            self._client = None
            self._config_fingerprint = None

        if client is not None:
            await client.close()

    async def aclose(self) -> None:
        await self.reset()


__all__ = ["EmbeddingClientProvider", "ResolvedEmbeddingConfig"]
