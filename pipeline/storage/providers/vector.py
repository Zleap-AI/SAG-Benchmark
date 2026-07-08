"""Vector/search backend implementations."""

import json
import math
from typing import Any, Dict, List, Optional

from elasticsearch_dsl import Q
from sqlalchemy import bindparam, text

from pipeline.core.config import get_settings
from pipeline.core.storage.elasticsearch import close_es_client, get_es_client
from pipeline.db.base import get_engine
from pipeline.storage.capabilities import StorageCapabilities
from pipeline.utils import get_logger

logger = get_logger("storage.vector")


def _bulk_result(total: int, indexed: int, failed: int = 0) -> Dict[str, Any]:
    return {
        "total": total,
        "indexed": indexed,
        "failed": failed,
        "success": failed == 0,
    }


def _clean_vector(vector: Optional[List[float]]) -> Optional[List[float]]:
    if vector is None:
        return None
    cleaned: List[float] = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("vector contains NaN or Inf")
        cleaned.append(number)
    return cleaned


def _serialize_vector(vector: List[float]) -> str:
    cleaned = _clean_vector(vector)
    if cleaned is None:
        raise ValueError("vector is required")
    return "[" + ",".join(format(value, ".9g") for value in cleaned) + "]"


def _score_from_cosine_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    value = max(0.0, min(2.0, value))
    return 1.0 - (value / 2.0)


def _ef_search_literal() -> int:
    value = int(get_settings().oceanbase_vector_search_ef_search)
    if value < 1 or value > 160000:
        raise ValueError("OCEANBASE_VECTOR_SEARCH_EF_SEARCH must be between 1 and 160000")
    return value


class ElasticsearchVectorStore:
    """Elasticsearch-backed vector/search store."""

    backend_name = "elasticsearch"

    async def health_check(self) -> bool:
        return await get_es_client().ping()

    def capabilities(self, database_backend: str) -> StorageCapabilities:
        settings = get_settings()
        return StorageCapabilities(
            database_backend=database_backend,
            vector_backend=self.backend_name,
            supports_vector_search=True,
            supports_fulltext_search=True,
            supports_hybrid_search=True,
            supports_transactional_vector_write=False,
            max_vector_dims=settings.embedding_dimensions,
        )

    async def close(self) -> None:
        await close_es_client()

    async def _bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        routing: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        es_client = get_es_client()
        indexed = failed = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            result = await es_client.bulk_index(
                index=index_name,
                documents=batch,
                return_details=True,
                routing=routing,
            )
            indexed += int(result["success_count"])
            failed += int(result["error_count"])
        return _bulk_result(len(documents), indexed, failed)

    async def upsert_chunk_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return await self._bulk_index(
            "source_chunks",
            documents,
            routing=kwargs.get("routing"),
            batch_size=kwargs.get("batch_size", 50),
        )

    async def upsert_event_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return await self._bulk_index(
            "event_vectors",
            documents,
            routing=kwargs.get("routing"),
            batch_size=kwargs.get("batch_size", 50),
        )

    async def upsert_entity_vectors(self, documents: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return await self._bulk_index(
            "entity_vectors",
            documents,
            routing=kwargs.get("routing"),
            batch_size=kwargs.get("batch_size", 50),
        )

    async def upsert_event_entity_vectors(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        return await self._bulk_index(
            "event_entity_vectors",
            documents,
            routing=kwargs.get("routing"),
            batch_size=kwargs.get("batch_size", 50),
        )

    async def search_chunks_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        chunk_type: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        filters = []
        if source_config_ids:
            filters.append(Q("terms", source_config_id=source_config_ids))
        elif source_config_id:
            filters.append(Q("term", source_config_id=source_config_id))
        if chunk_type:
            filters.append(Q("term", chunk_type=chunk_type))
        if chunk_ids:
            filters.append(Q("terms", _id=chunk_ids))

        filter_query = Q("bool", must=filters).to_dict() if filters else None
        routing = source_config_id if source_config_id else None
        return await get_es_client().vector_search(
            index="source_chunks",
            field="content_vector",
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )

    async def search_events_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        vector_field: str = "content_vector",
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        filters = []
        if source_config_ids:
            filters.append(Q("terms", source_config_id=source_config_ids))
        elif source_config_id:
            filters.append(Q("term", source_config_id=source_config_id))
        if event_ids:
            filters.append(Q("terms", event_id=event_ids))

        filter_query = Q("bool", must=filters).to_dict() if filters else None
        routing = source_config_id if source_config_id else None
        return await get_es_client().vector_search(
            index="event_vectors",
            field=vector_field,
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )

    async def search_entities_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        filters = []
        if source_config_ids:
            filters.append(Q("terms", source_config_id=source_config_ids))
        elif source_config_id:
            filters.append(Q("term", source_config_id=source_config_id))
        if entity_type:
            filters.append(Q("term", type=entity_type))

        filter_query = Q("bool", must=filters).to_dict() if filters else None
        routing = source_config_id if source_config_id else None
        return await get_es_client().vector_search(
            index="entity_vectors",
            field="vector",
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )

    async def search_event_entities_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        event_ids: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        filters = [Q("term", is_delete=False)]
        if source_config_ids:
            filters.append(Q("terms", source_config_id=source_config_ids))
        elif source_config_id:
            filters.append(Q("term", source_config_id=source_config_id))
        if event_ids:
            filters.append(Q("terms", event_id=event_ids))
        elif event_id:
            filters.append(Q("term", event_id=event_id))
        if entity_ids:
            filters.append(Q("terms", entity_id=entity_ids))
        elif entity_id:
            filters.append(Q("term", entity_id=entity_id))

        filter_query = Q("bool", must=filters).to_dict()
        routing = source_config_id if source_config_id else None
        return await get_es_client().vector_search(
            index="event_entity_vectors",
            field="vector",
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )


class OceanBaseVectorStore:
    """OceanBase-backed vector/search store."""

    backend_name = "oceanbase"

    async def health_check(self) -> bool:
        try:
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("OceanBase vector health check failed: %s", exc)
            return False

    def capabilities(self, database_backend: str) -> StorageCapabilities:
        settings = get_settings()
        return StorageCapabilities(
            database_backend=database_backend,
            vector_backend=self.backend_name,
            supports_vector_search=True,
            supports_fulltext_search=True,
            supports_hybrid_search=True,
            supports_transactional_vector_write=True,
            max_vector_dims=settings.embedding_dimensions,
        )

    async def close(self) -> None:
        return None

    async def upsert_chunk_vectors(self, documents: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        if not documents:
            return _bulk_result(0, 0)

        stmt = text(
            """
            UPDATE source_chunk
            SET heading_vector = :heading_vector,
                content_vector = :content_vector
            WHERE id = :chunk_id
            """
        )
        indexed = failed = 0
        engine = get_engine()
        async with engine.begin() as conn:
            for doc in documents:
                try:
                    await conn.execute(
                        stmt,
                        {
                            "chunk_id": doc.get("chunk_id") or doc.get("id"),
                            "heading_vector": (
                                _serialize_vector(doc["heading_vector"])
                                if doc.get("heading_vector") is not None
                                else None
                            ),
                            "content_vector": _serialize_vector(doc["content_vector"]),
                        },
                    )
                    indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "OceanBase chunk vector upsert failed: id=%s error=%s",
                        doc.get("chunk_id") or doc.get("id"),
                        exc,
                    )
        return _bulk_result(len(documents), indexed, failed)

    async def upsert_event_vectors(self, documents: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        if not documents:
            return _bulk_result(0, 0)

        stmt = text(
            """
            UPDATE source_event
            SET title_vector = :title_vector,
                content_vector = :content_vector,
                entities = :entities
            WHERE id = :event_id
            """
        )
        indexed = failed = 0
        engine = get_engine()
        async with engine.begin() as conn:
            for doc in documents:
                try:
                    await conn.execute(
                        stmt,
                        {
                            "event_id": doc.get("event_id") or doc.get("id"),
                            "title_vector": _serialize_vector(doc["title_vector"]),
                            "content_vector": _serialize_vector(doc["content_vector"]),
                            "entities": json.dumps(doc.get("entities") or [], ensure_ascii=False),
                        },
                    )
                    indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "OceanBase event vector upsert failed: id=%s error=%s",
                        doc.get("event_id") or doc.get("id"),
                        exc,
                    )
        return _bulk_result(len(documents), indexed, failed)

    async def upsert_entity_vectors(self, documents: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        if not documents:
            return _bulk_result(0, 0)

        stmt = text(
            """
            UPDATE entity
            SET vector = :vector
            WHERE id = :entity_id
            """
        )
        indexed = failed = 0
        engine = get_engine()
        async with engine.begin() as conn:
            for doc in documents:
                try:
                    await conn.execute(
                        stmt,
                        {
                            "entity_id": doc.get("entity_id") or doc.get("id"),
                            "vector": _serialize_vector(doc["vector"]),
                        },
                    )
                    indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "OceanBase entity vector upsert failed: id=%s error=%s",
                        doc.get("entity_id") or doc.get("id"),
                        exc,
                    )
        return _bulk_result(len(documents), indexed, failed)

    async def upsert_event_entity_vectors(
        self, documents: List[Dict[str, Any]], **_: Any
    ) -> Dict[str, Any]:
        if not documents:
            return _bulk_result(0, 0)

        stmt = text(
            """
            UPDATE event_entity
            SET vector = :vector
            WHERE id = :association_id
            """
        )
        indexed = failed = 0
        engine = get_engine()
        async with engine.begin() as conn:
            for doc in documents:
                try:
                    await conn.execute(
                        stmt,
                        {
                            "association_id": (
                                doc.get("association_id")
                                or doc.get("event_entity_id")
                                or doc.get("id")
                            ),
                            "vector": _serialize_vector(doc["vector"]),
                        },
                    )
                    indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "OceanBase event-entity vector upsert failed: id=%s error=%s",
                        doc.get("association_id")
                        or doc.get("event_entity_id")
                        or doc.get("id"),
                        exc,
                    )
        return _bulk_result(len(documents), indexed, failed)

    async def search_chunks_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        exact: bool = False,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "query_vector": _serialize_vector(query_vector),
            "top_k": k,
        }
        where = ["content_vector IS NOT NULL"]
        bindparams = []
        if source_config_ids:
            where.append("source_config_id IN :source_config_ids")
            params["source_config_ids"] = source_config_ids
            bindparams.append(bindparam("source_config_ids", expanding=True))
        elif source_config_id:
            where.append("source_config_id = :source_config_id")
            params["source_config_id"] = source_config_id
        if chunk_ids:
            where.append("id IN :chunk_ids")
            params["chunk_ids"] = chunk_ids
            bindparams.append(bindparam("chunk_ids", expanding=True))

        ef_search = _ef_search_literal()
        knn_suffix = (
            "LIMIT :top_k"
            if exact
            else f"APPROX LIMIT :top_k PARAMETERS (ef_search={ef_search})"
        )
        sql = text(
            f"""
            SELECT id AS chunk_id,
                   source_id,
                   source_config_id,
                   rank,
                   heading,
                   content,
                   chunk_length AS content_length,
                   `references`,
                   cosine_distance(content_vector, :query_vector) AS distance
            FROM source_chunk
            WHERE {" AND ".join(where)}
            ORDER BY cosine_distance(content_vector, :query_vector)
            {knn_suffix}
            """
        )
        if bindparams:
            sql = sql.bindparams(*bindparams)

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [_with_score(row) for row in rows]

    async def search_events_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        vector_field: str = "content_vector",
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        exact: bool = False,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        if vector_field not in {"title_vector", "content_vector"}:
            raise ValueError(f"unsupported event vector field: {vector_field}")

        params: Dict[str, Any] = {
            "query_vector": _serialize_vector(query_vector),
            "top_k": k,
        }
        where = [f"{vector_field} IS NOT NULL", "(status IS NULL OR status != 'DELETED')"]
        bindparams = []
        if source_config_ids:
            where.append("source_config_id IN :source_config_ids")
            params["source_config_ids"] = source_config_ids
            bindparams.append(bindparam("source_config_ids", expanding=True))
        elif source_config_id:
            where.append("source_config_id = :source_config_id")
            params["source_config_id"] = source_config_id
        if event_ids:
            where.append("id IN :event_ids")
            params["event_ids"] = event_ids
            bindparams.append(bindparam("event_ids", expanding=True))

        ef_search = _ef_search_literal()
        knn_suffix = (
            "LIMIT :top_k"
            if exact
            else f"APPROX LIMIT :top_k PARAMETERS (ef_search={ef_search})"
        )
        sql = text(
            f"""
            SELECT id AS event_id,
                   source_config_id,
                   source_type,
                   source_id,
                   chunk_id,
                   title,
                   summary,
                   content,
                   entities,
                   start_time,
                   end_time,
                   cosine_distance({vector_field}, :query_vector) AS distance
            FROM source_event
            WHERE {" AND ".join(where)}
            ORDER BY cosine_distance({vector_field}, :query_vector)
            {knn_suffix}
            """
        )
        if bindparams:
            sql = sql.bindparams(*bindparams)

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [_with_score(row) for row in rows]

    async def search_entities_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        exact: bool = False,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "query_vector": _serialize_vector(query_vector),
            "top_k": k,
        }
        where = ["vector IS NOT NULL"]
        bindparams = []
        if source_config_ids:
            where.append("source_config_id IN :source_config_ids")
            params["source_config_ids"] = source_config_ids
            bindparams.append(bindparam("source_config_ids", expanding=True))
        elif source_config_id:
            where.append("source_config_id = :source_config_id")
            params["source_config_id"] = source_config_id
        if entity_type:
            where.append("type = :entity_type")
            params["entity_type"] = entity_type

        ef_search = _ef_search_literal()
        knn_suffix = (
            "LIMIT :top_k"
            if exact
            else f"APPROX LIMIT :top_k PARAMETERS (ef_search={ef_search})"
        )
        sql = text(
            f"""
            SELECT id AS entity_id,
                   source_config_id,
                   type,
                   name,
                   normalized_name,
                   description,
                   cosine_distance(vector, :query_vector) AS distance
            FROM entity
            WHERE {" AND ".join(where)}
            ORDER BY cosine_distance(vector, :query_vector)
            {knn_suffix}
            """
        )
        if bindparams:
            sql = sql.bindparams(*bindparams)

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [_with_score(row) for row in rows]

    async def search_event_entities_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        event_ids: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        exact: bool = False,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "query_vector": _serialize_vector(query_vector),
            "top_k": k,
        }
        where = ["ee.vector IS NOT NULL", "(se.status IS NULL OR se.status != 'DELETED')"]
        bindparams = []
        if source_config_ids:
            where.append("se.source_config_id IN :source_config_ids")
            params["source_config_ids"] = source_config_ids
            bindparams.append(bindparam("source_config_ids", expanding=True))
        elif source_config_id:
            where.append("se.source_config_id = :source_config_id")
            params["source_config_id"] = source_config_id
        if event_ids:
            where.append("ee.event_id IN :event_ids")
            params["event_ids"] = event_ids
            bindparams.append(bindparam("event_ids", expanding=True))
        elif event_id:
            where.append("ee.event_id = :event_id")
            params["event_id"] = event_id
        if entity_ids:
            where.append("ee.entity_id IN :entity_ids")
            params["entity_ids"] = entity_ids
            bindparams.append(bindparam("entity_ids", expanding=True))
        elif entity_id:
            where.append("ee.entity_id = :entity_id")
            params["entity_id"] = entity_id

        ef_search = _ef_search_literal()
        knn_suffix = (
            "LIMIT :top_k"
            if exact
            else f"APPROX LIMIT :top_k PARAMETERS (ef_search={ef_search})"
        )
        sql = text(
            f"""
            SELECT ee.id AS association_id,
                   ee.event_id,
                   ee.entity_id,
                   se.source_config_id,
                   ee.description,
                   cosine_distance(ee.vector, :query_vector) AS distance
            FROM event_entity ee
            JOIN source_event se ON se.id = ee.event_id
            WHERE {" AND ".join(where)}
            ORDER BY cosine_distance(ee.vector, :query_vector)
            {knn_suffix}
            """
        )
        if bindparams:
            sql = sql.bindparams(*bindparams)

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [_with_score(row) for row in rows]


def _with_score(row: Any) -> Dict[str, Any]:
    item = dict(row)
    item["_score"] = _score_from_cosine_distance(item.get("distance"))
    return item
