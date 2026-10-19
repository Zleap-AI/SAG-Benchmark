"""Higher-level retrieval providers.

This layer intentionally sits above the raw vector store. Multi-hop search needs
text recall, graph expansion, event fetches, and vector KNN; only the provider
should care whether those operations are backed by Elasticsearch or OceanBase.
"""

from typing import Any

from sqlalchemy import bindparam, or_, select, text
from sqlalchemy.exc import SQLAlchemyError

from pipeline.db import Entity, EventEntity, SourceEvent, get_session_factory
from pipeline.storage.backends.elasticsearch.client import get_es_client
from pipeline.storage.backends.elasticsearch.repositories.entity_repository import (
    EntityVectorRepository,
)
from pipeline.storage.backends.elasticsearch.repositories.event_entity_repository import (
    EventEntityRepository,
)
from pipeline.storage.backends.elasticsearch.repositories.event_repository import (
    EventVectorRepository,
)
from pipeline.storage.interfaces import VectorSearchStore
from pipeline.utils import get_logger

logger = get_logger("storage.search")


class ElasticsearchSearchStore:
    """High-level retrieval API backed by Elasticsearch indices."""

    backend_name = "elasticsearch"

    def __init__(self, vector_store: VectorSearchStore) -> None:
        self._vector_store = vector_store
        es_client = get_es_client()
        self._entity_repo = EntityVectorRepository(es_client)
        self._event_repo = EventVectorRepository(es_client)
        self._event_entity_repo = EventEntityRepository(es_client)

    async def close(self) -> None:
        """Elasticsearch lifecycle is owned by the vector store."""
        return None

    async def search_entities_by_text(
        self,
        query: str,
        source_config_ids: list[str] | None = None,
        size: int = 20,
        entity_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._entity_repo.search_by_query_bm25(
            query=query,
            source_config_ids=source_config_ids,
            size=size,
            allowed_entity_ids=entity_ids,
        )

    async def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        return await self._entity_repo.get_entities_by_ids(entity_ids)

    async def get_event_ids_by_entity_ids(
        self,
        *,
        entity_ids: list[str],
        source_config_ids: list[str] | None = None,
        exclude_event_ids: list[str] | None = None,
        size: int = 100,
    ) -> list[str]:
        return await self._event_entity_repo.get_event_ids_by_entity_ids(
            entity_ids=entity_ids,
            source_config_ids=source_config_ids,
            exclude_event_ids=exclude_event_ids,
            size=size,
        )

    async def get_events_by_ids(
        self,
        event_ids: list[str],
        source_includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._event_repo.get_events_by_ids(
            event_ids,
            source_includes=source_includes,
        )

    async def search_events_by_text(
        self,
        *,
        query: str,
        event_ids: list[str],
        k: int = 100,
        source_config_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._event_repo.search_bm25_by_text(
            query=query,
            event_ids=event_ids,
            k=k,
            source_config_ids=source_config_ids,
        )

    async def search_events_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int = 10,
        source_config_ids: list[str] | None = None,
        event_ids: list[str] | None = None,
        vector_field: str = "content_vector",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await self._vector_store.search_events_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
            vector_field=vector_field,
            **kwargs,
        )

    async def search_chunks_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int = 10,
        source_config_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await self._vector_store.search_chunks_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
            **kwargs,
        )


class OceanBaseSearchStore:
    """High-level retrieval API backed by OceanBase SQL + vector columns."""

    backend_name = "oceanbase"

    def __init__(self, vector_store: VectorSearchStore) -> None:
        self._vector_store = vector_store

    async def close(self) -> None:
        """OceanBase SQL lifecycle is owned by the database store."""
        return None

    async def search_entities_by_text(
        self,
        query: str,
        source_config_ids: list[str] | None = None,
        size: int = 20,
        entity_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_text = (query or "").strip()
        if not query_text or size <= 0:
            return []

        try:
            return await self._search_entities_by_fulltext(
                query_text,
                source_config_ids=source_config_ids,
                size=size,
                entity_ids=entity_ids,
            )
        except SQLAlchemyError as exc:
            logger.warning(
                "OceanBase fulltext entity search failed; falling back to LIKE: %s",
                exc,
            )
            return await self._search_entities_by_like(
                query_text,
                source_config_ids=source_config_ids,
                size=size,
                entity_ids=entity_ids,
            )

    async def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        stmt = select(Entity).where(Entity.id.in_(entity_ids))
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            entities = result.scalars().all()
        return [
            {
                "entity_id": entity.id,
                "source_config_id": entity.source_config_id,
                "type": entity.type,
                "name": entity.name,
                "normalized_name": entity.normalized_name,
                "description": entity.description,
            }
            for entity in entities
        ]

    async def _search_entities_by_fulltext(
        self,
        query: str,
        *,
        source_config_ids: list[str] | None,
        size: int,
        entity_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "size": size,
        }
        where = ["MATCH(name, normalized_name) AGAINST (:query IN NATURAL LANGUAGE MODE)"]
        bindparams = []
        if source_config_ids:
            where.append("source_config_id IN :source_config_ids")
            params["source_config_ids"] = source_config_ids
            bindparams.append(bindparam("source_config_ids", expanding=True))
        if entity_ids:
            where.append("id IN :entity_ids")
            params["entity_ids"] = entity_ids
            bindparams.append(bindparam("entity_ids", expanding=True))

        stmt = text(f"""
            SELECT id AS entity_id,
                   source_config_id,
                   type,
                   name,
                   normalized_name,
                   description,
                   MATCH(name, normalized_name) AGAINST (:query IN NATURAL LANGUAGE MODE) AS score
            FROM entity
            WHERE {" AND ".join(where)}
            ORDER BY score DESC
            LIMIT :size
            """)
        if bindparams:
            stmt = stmt.bindparams(*bindparams)

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt, params)
            rows = result.mappings().all()

        return [
            {
                **dict(row),
                "_score": float(row.get("score") or 0.0),
            }
            for row in rows
        ]

    async def _search_entities_by_like(
        self,
        query: str,
        *,
        source_config_ids: list[str] | None,
        size: int,
        entity_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        tokens: list[str] = []
        for token in query.split():
            if len(token) >= 2 and token not in tokens:
                tokens.append(token)
        if query not in tokens:
            tokens.insert(0, query)
        tokens = tokens[:8]

        conditions = []
        for token in tokens:
            pattern = f"%{token}%"
            conditions.extend(
                [
                    Entity.name.like(pattern),
                    Entity.normalized_name.like(pattern),
                ]
            )

        stmt = (
            select(
                Entity.id,
                Entity.source_config_id,
                Entity.type,
                Entity.name,
                Entity.normalized_name,
                Entity.description,
            )
            .where(or_(*conditions))
            .limit(size)
        )
        if source_config_ids:
            stmt = stmt.where(Entity.source_config_id.in_(source_config_ids))
        if entity_ids:
            stmt = stmt.where(Entity.id.in_(entity_ids))

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [
            {
                "entity_id": row[0],
                "source_config_id": row[1],
                "type": row[2],
                "name": row[3],
                "normalized_name": row[4],
                "description": row[5],
                "_score": 1.0,
            }
            for row in rows
        ]

    async def get_event_ids_by_entity_ids(
        self,
        *,
        entity_ids: list[str],
        source_config_ids: list[str] | None = None,
        exclude_event_ids: list[str] | None = None,
        size: int = 100,
    ) -> list[str]:
        if not entity_ids:
            return []

        stmt = (
            select(EventEntity.event_id)
            .join(SourceEvent, EventEntity.event_id == SourceEvent.id)
            .where(EventEntity.entity_id.in_(entity_ids))
            .limit(size)
        )
        if source_config_ids:
            stmt = stmt.where(SourceEvent.source_config_id.in_(source_config_ids))
        if exclude_event_ids:
            stmt = stmt.where(EventEntity.event_id.not_in(exclude_event_ids))

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        event_ids_out: list[str] = []
        seen: set = set()
        for row in rows:
            event_id = row[0]
            if event_id and event_id not in seen:
                seen.add(event_id)
                event_ids_out.append(event_id)
        return event_ids_out

    async def get_events_by_ids(
        self,
        event_ids: list[str],
        source_includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not event_ids:
            return []

        includes = source_includes or [
            "source_config_id",
            "source_type",
            "source_id",
            "chunk_id",
            "title",
            "summary",
            "content",
            "entities",
            "start_time",
            "end_time",
        ]
        stmt = select(SourceEvent).where(SourceEvent.id.in_(event_ids))
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            events = result.scalars().all()

            event_entity_map: dict[str, list[str]] = {}
            if "entity_ids" in includes:
                rel_stmt = select(EventEntity.event_id, EventEntity.entity_id).where(
                    EventEntity.event_id.in_(event_ids)
                )
                rel_result = await session.execute(rel_stmt)
                for event_id, entity_id in rel_result.fetchall():
                    event_entity_map.setdefault(event_id, []).append(entity_id)

        rows: list[dict[str, Any]] = []
        for event in events:
            row: dict[str, Any] = {"event_id": event.id}
            for field in includes:
                if field == "event_id":
                    continue
                if field == "entity_ids":
                    row[field] = event_entity_map.get(event.id, [])
                elif field == "entities" and hasattr(event, "entity_ids_json"):
                    row[field] = event.entity_ids_json
                else:
                    row[field] = getattr(event, field, None)
            rows.append(row)
        return rows

    async def search_events_by_text(
        self,
        *,
        query: str,
        event_ids: list[str],
        k: int = 100,
        source_config_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip() or not event_ids or k <= 0:
            return []
        stmt = select(SourceEvent).where(SourceEvent.id.in_(event_ids))
        if source_config_ids:
            stmt = stmt.where(SourceEvent.source_config_id.in_(source_config_ids))
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(stmt)
            events = result.scalars().all()

        tokens = [token.casefold() for token in query.split() if token.strip()]
        if not tokens:
            tokens = [query.casefold()]
        ranked: list[dict[str, Any]] = []
        for event in events:
            title = str(event.title or "")
            summary = str(event.summary or "")
            content = str(event.content or "")
            searchable = f"{title} {summary} {content}".casefold()
            score = float(sum(searchable.count(token) for token in tokens))
            if score <= 0:
                continue
            ranked.append(
                {
                    "event_id": event.id,
                    "title": title,
                    "summary": summary,
                    "content": content,
                    "_score": score,
                }
            )
        ranked.sort(key=lambda item: (-float(item["_score"]), str(item["event_id"])))
        return ranked[:k]

    async def search_events_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int = 10,
        source_config_ids: list[str] | None = None,
        event_ids: list[str] | None = None,
        vector_field: str = "content_vector",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        kwargs.setdefault("exact", bool(event_ids))
        return await self._vector_store.search_events_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
            vector_field=vector_field,
            **kwargs,
        )

    async def search_chunks_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int = 10,
        source_config_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await self._vector_store.search_chunks_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
            **kwargs,
        )
