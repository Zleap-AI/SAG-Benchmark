"""SQL contract tests for the event-universe storage boundary."""

from types import SimpleNamespace

import pytest

from pipeline.storage.providers import database as database_provider


class _FakeResult:
    def __init__(self, *, rows=(), scalar_rows=()):
        self._rows = list(rows)
        self._scalar_rows = list(scalar_rows)

    def fetchall(self):
        return self._rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self._scalar_rows)


class _FakeSession:
    def __init__(self, results):
        self._results = iter(results)
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def execute(self, statement):
        self.statements.append(statement)
        return next(self._results)


def _install_session(monkeypatch, *results):
    session = _FakeSession(results)
    monkeypatch.setattr(
        database_provider,
        "get_session_factory",
        lambda: lambda: session,
    )
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_class",
    [
        database_provider.MySqlDatabaseStore,
        database_provider.OceanBaseDatabaseStore,
    ],
)
async def test_filter_active_event_ids_preserves_input_order_and_sql_scope(
    monkeypatch,
    store_class,
):
    session = _install_session(
        monkeypatch,
        _FakeResult(rows=[("event-2",), ("event-1",)]),
    )

    result = await store_class().filter_active_event_ids(
        ["event-1", "event-2", "event-3"],
        source_config_ids=["source-1"],
    )

    assert result == ["event-1", "event-2"]
    sql = str(session.statements[0])
    assert "FROM source_event" in sql
    assert "source_event.status IS NULL OR source_event.status !=" in sql
    assert "source_event.source_config_id IN" in sql


@pytest.mark.asyncio
async def test_scope_relation_sql_has_active_filter_order_and_global_limit(monkeypatch):
    session = _install_session(
        monkeypatch,
        _FakeResult(rows=[("event-1", "entity-1"), ("event-2", "entity-2")]),
    )

    result = await database_provider.MySqlDatabaseStore().get_event_entity_pairs_by_events(
        ["event-1", "event-2"],
        source_config_ids=["source-1"],
        limit=2,
    )

    assert result == [("event-1", "entity-1"), ("event-2", "entity-2")]
    sql = str(session.statements[0])
    assert "JOIN source_event ON source_event.id = event_entity.event_id" in sql
    assert "source_event.status IS NULL OR source_event.status !=" in sql
    assert "source_event.source_config_id IN" in sql
    assert "ORDER BY event_entity.event_id, event_entity.entity_id" in sql
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_entity_recall_sql_keeps_existing_no_active_filter_semantics(monkeypatch):
    session = _install_session(
        monkeypatch,
        _FakeResult(rows=[("event-1", "entity-1")]),
    )

    result = await database_provider.MySqlDatabaseStore().get_event_entity_pairs_by_entities(
        ["entity-1"],
        source_config_ids=["source-1"],
    )

    assert result == [("event-1", "entity-1")]
    sql = str(session.statements[0])
    assert "JOIN source_event ON source_event.id = event_entity.event_id" in sql
    assert "source_event.source_config_id IN" in sql
    assert "source_event.status" not in sql


@pytest.mark.asyncio
async def test_entity_recall_without_source_scope_skips_source_event_join(monkeypatch):
    session = _install_session(
        monkeypatch,
        _FakeResult(rows=[("event-1", "entity-1")]),
    )

    await database_provider.MySqlDatabaseStore().get_event_entity_pairs_by_entities(
        ["entity-1"],
    )

    sql = str(session.statements[0])
    assert "JOIN source_event" not in sql
    assert "source_event.source_config_id" not in sql


@pytest.mark.asyncio
async def test_chunk_hydration_keeps_sag2_payload_shape(monkeypatch):
    chunk = SimpleNamespace(
        id="chunk-1",
        source_id=None,
        source_config_id="source-1",
        heading=None,
        content="chunk body",
        rank=3,
    )
    session = _install_session(
        monkeypatch,
        _FakeResult(rows=[("event-1", "chunk-1"), ("event-2", None)]),
        _FakeResult(scalar_rows=[chunk]),
    )

    result = await database_provider.MySqlDatabaseStore().get_chunks_by_event_ids(
        ["event-1", "event-2"],
    )

    assert result == {
        "event-1": {
            "chunk_id": "chunk-1",
            "source_id": "",
            "source_config_id": "source-1",
            "heading": "",
            "content": "chunk body",
            "rank": 3,
        }
    }
    assert len(session.statements) == 2
    assert "FROM source_event" in str(session.statements[0])
    assert "FROM source_chunk" in str(session.statements[1])
