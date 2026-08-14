import pytest

from pipeline.core.ai import factory


class FakeProvider:
    def __init__(self) -> None:
        self.shared = object()
        self.owned = object()

    async def get(self, scenario: str):
        assert scenario == "general"
        return self.shared

    async def create_owned(self, scenario: str, overrides):
        assert scenario == "general"
        assert overrides == {"model": "owned"}
        return self.owned


@pytest.mark.asyncio
async def test_factory_wrappers_do_not_require_runtime_forward_reference(monkeypatch) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(factory, "_embedding_provider", provider)

    assert await factory.get_embedding_client() is provider.shared
    assert (
        await factory.create_embedding_client(embedding_config={"model": "owned"}) is provider.owned
    )
