import pytest

from pipeline.modules.search.config import SAGConfig
from pipeline.modules.search.prompts import PromptProvider
from pipeline.modules.search.sag2.candidate_scope import SAG2CandidatePoolBuilder
from pipeline.modules.search.sag2.recall import SAG2RecallStage
from pipeline.modules.search.sag2.runtime import SAG2Runtime
from pipeline.modules.search.sag2.timing import SAG2TimingService
from pipeline.storage.providers import database as database_provider


class FakeEventUniverseStore:
    def __init__(self):
        self.calls = []

    async def filter_active_event_ids(self, event_ids, source_config_ids=None):
        self.calls.append(("active", list(event_ids), source_config_ids))
        return [event_id for event_id in event_ids if event_id != "deleted"]

    async def get_event_entity_pairs_by_events(self, event_ids, source_config_ids=None, limit=None):
        self.calls.append(("scope_pairs", list(event_ids), source_config_ids, limit))
        return [("e1", "x"), ("e2", "y")]

    async def get_event_entity_pairs_by_entities(self, entity_ids, source_config_ids=None):
        self.calls.append(("entity_pairs", list(entity_ids), source_config_ids))
        return [("e1", "x"), ("e2", "x"), ("e3", "x"), ("e4", "y")]

    async def get_chunks_by_event_ids(self, event_ids):
        return {}


@pytest.mark.asyncio
async def test_scope_builder_keeps_algorithm_in_memory_and_uses_store_only_for_rows():
    store = FakeEventUniverseStore()

    async def event_search(**_kwargs):
        return [
            {"event_id": "e1", "_score": 0.9},
            {"event_id": "deleted", "_score": 0.8},
            {"event_id": "e2", "_score": 0.7},
        ]

    scope = await SAG2CandidatePoolBuilder(event_search, store).build(
        query_vector=[0.1],
        source_config_ids=["source-1"],
        event_top_k=3,
        bootstrap_entity_limit=2,
    )

    assert list(scope.event_scores) == ["e1", "e2"]
    assert scope.event_to_entities == {"e1": ["x"], "e2": ["y"]}
    assert scope.entity_to_events == {"x": ["e1"], "y": ["e2"]}
    assert store.calls == [
        ("active", ["e1", "deleted", "e2"], ["source-1"]),
        ("scope_pairs", ["e1", "e2"], ["source-1"], 2),
    ]


@pytest.mark.asyncio
async def test_entity_recall_keeps_exclusion_and_per_entity_limit_in_sag2():
    store = FakeEventUniverseStore()
    runtime = SAG2Runtime.__new__(SAG2Runtime)
    runtime._event_universe_store = store
    stage = SAG2RecallStage(
        runtime=runtime,
        timing=SAG2TimingService(runtime, ctx_var_name="test_sag2_store_timing"),
    )

    mapping, event_ids = await stage.events_from_entities(
        entity_ids=["x", "y"],
        source_config_ids=["source-1"],
        config=SAGConfig(sag2_recall={"max_events_per_key": 2}),
        exclude_event_ids={"e1"},
    )

    assert mapping == {"e2": ["x"], "e3": ["x"], "e4": ["y"]}
    assert event_ids == ["e2", "e3", "e4"]
    assert store.calls == [("entity_pairs", ["x", "y"], ["source-1"])]


def test_sag2_prompt_provider_keeps_message_shapes_and_variables():
    provider = PromptProvider(use_mlflow=False)

    rewrite = provider.get_sag2_rewrite_messages(
        query="Who wrote this?",
        current_timestamp="2026-08-14 00:00:00 UTC",
        max_entities=7,
    )
    assert [message.role.value for message in rewrite] == ["system", "user"]
    assert "Who wrote this?" in rewrite[1].content
    assert "Max entities: 7" in rewrite[1].content

    rerank = provider.get_sag2_rerank_messages(
        question="Who wrote this?",
        relations="[0] author",
        top_k=1,
    )
    assert [message.role.value for message in rerank] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Select exactly 1 relationships" in rerank[0].content
    assert rerank[-1].content.endswith("[0] author")


@pytest.mark.asyncio
async def test_runtime_aclose_only_closes_its_llm_client():
    closed = []

    class ClosableClient:
        async def close(self):
            closed.append(True)

    runtime = SAG2Runtime.__new__(SAG2Runtime)
    runtime._llm_client = ClosableClient()
    runtime._processor = object()
    runtime._prompts = object()

    await runtime.aclose()

    assert closed == [True]
    assert runtime._llm_client is None
    assert runtime._processor is None
    assert runtime._prompts is None


@pytest.mark.asyncio
async def test_database_event_universe_provider_has_session_factory_dependency(monkeypatch):
    class FakeResult:
        @staticmethod
        def fetchall():
            return [("event-1", "entity-1")]

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, _statement):
            return FakeResult()

    monkeypatch.setattr(
        database_provider,
        "get_session_factory",
        lambda: lambda: FakeSession(),
    )

    rows = await database_provider.MySQLDatabaseStore().get_event_entity_pairs_by_entities(
        ["entity-1"]
    )

    assert rows == [("event-1", "entity-1")]
