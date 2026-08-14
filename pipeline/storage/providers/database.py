"""Structured database backend and event-universe implementations."""

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text

from pipeline.db.base import close_database, get_engine, get_session_factory
from pipeline.db import EventEntity, SourceChunk, SourceEvent


class _SqlAlchemyDatabaseStore:
    backend_name = "sqlalchemy"

    async def health_check(self) -> bool:
        try:
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await close_database()

    async def filter_active_event_ids(
        self,
        event_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Filter candidate ids using the canonical SourceEvent semantics."""
        normalized = [str(event_id) for event_id in event_ids if event_id]
        if not normalized:
            return []

        filters = [SourceEvent.id.in_(normalized), SourceEvent.not_deleted()]
        if source_config_ids:
            filters.append(SourceEvent.source_config_id.in_(source_config_ids))

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(select(SourceEvent.id).where(*filters))
            valid_ids = {str(row[0]) for row in result.fetchall()}

        # Preserve the vector-recall order used by SAG2CandidatePoolBuilder.
        return [event_id for event_id in normalized if event_id in valid_ids]

    async def get_event_entity_pairs_by_events(
        self,
        event_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """Read scope edges with the existing active/source/order semantics."""
        normalized = [str(event_id) for event_id in event_ids if event_id]
        if not normalized:
            return []

        filters = [EventEntity.event_id.in_(normalized), SourceEvent.not_deleted()]
        if source_config_ids:
            filters.append(SourceEvent.source_config_id.in_(source_config_ids))
        stmt = (
            select(EventEntity.event_id, EventEntity.entity_id)
            .join(SourceEvent, SourceEvent.id == EventEntity.event_id)
            .where(*filters)
            .order_by(EventEntity.event_id, EventEntity.entity_id)
        )
        if limit:
            stmt = stmt.limit(limit)

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            return [
                (str(event_id), str(entity_id))
                for event_id, entity_id in result.fetchall()
                if event_id and entity_id
            ]

    async def get_event_entity_pairs_by_entities(
        self,
        entity_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Read entity-recall edges without changing its current semantics."""
        normalized = [str(entity_id) for entity_id in entity_ids if entity_id]
        if not normalized:
            return []

        stmt = select(EventEntity.event_id, EventEntity.entity_id).where(
            EventEntity.entity_id.in_(normalized)
        )
        # Preserve the existing _events_from_entities behavior: this path
        # applies source filtering when requested but does not add a new
        # not_deleted predicate in the same refactor.
        if source_config_ids:
            stmt = stmt.join(SourceEvent, SourceEvent.id == EventEntity.event_id).where(
                SourceEvent.source_config_id.in_(source_config_ids)
            )

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            return [
                (str(event_id), str(entity_id))
                for event_id, entity_id in result.fetchall()
                if event_id and entity_id
            ]

    async def get_chunks_by_event_ids(
        self,
        event_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Hydrate chunks with the exact shape previously built in Runtime."""
        normalized = [str(event_id) for event_id in event_ids if event_id]
        if not normalized:
            return {}

        session_factory = get_session_factory()
        async with session_factory() as session:
            event_result = await session.execute(
                select(SourceEvent.id, SourceEvent.chunk_id).where(
                    SourceEvent.id.in_(normalized)
                )
            )
            event_chunk_map: Dict[str, str] = {}
            chunk_ids: set[str] = set()
            for event_id, chunk_id in event_result.fetchall():
                if chunk_id:
                    event_id = str(event_id)
                    chunk_id = str(chunk_id)
                    event_chunk_map[event_id] = chunk_id
                    chunk_ids.add(chunk_id)
            if not chunk_ids:
                return {}

            chunk_result = await session.execute(
                select(SourceChunk).where(SourceChunk.id.in_(chunk_ids))
            )
            chunk_map: Dict[str, Dict[str, Any]] = {}
            for chunk in chunk_result.scalars().all():
                chunk_map[str(chunk.id)] = {
                    "chunk_id": str(chunk.id),
                    "source_id": chunk.source_id or "",
                    "source_config_id": chunk.source_config_id or "",
                    "heading": chunk.heading or "",
                    "content": chunk.content or "",
                    "rank": chunk.rank,
                }
            return {
                event_id: chunk_map[chunk_id]
                for event_id, chunk_id in event_chunk_map.items()
                if chunk_id in chunk_map
            }


class MySQLDatabaseStore(_SqlAlchemyDatabaseStore):
    """MySQL structured data backend."""

    backend_name = "mysql"


class OceanBaseDatabaseStore(_SqlAlchemyDatabaseStore):
    """OceanBase structured data backend in MySQL-compatible mode."""

    backend_name = "oceanbase"


# Backward-compatible import name. New code should use ``MySQLDatabaseStore``.
MySqlDatabaseStore = MySQLDatabaseStore
