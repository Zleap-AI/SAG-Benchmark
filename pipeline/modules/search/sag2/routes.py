"""Route-index operations for SAG2 reverse evidence tracing."""

from __future__ import annotations

import uuid
from typing import Any

from .utils import merge_ids, route_node_key


class SAG2RouteTracker:
    """Request-scoped route operations with no search-strategy dispatch."""

    @staticmethod
    def generate_query_id(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    @classmethod
    def new_route_index(cls, query: str) -> dict[str, Any]:
        query_node_id = cls.generate_query_id(query)
        query_key = route_node_key("query", query_node_id)
        return {
            "query_text": query,
            "query_node_id": query_node_id,
            "query_key": query_key,
            "original_query_key": query_key,
            "incoming": {},
            "event_scores": {},
            "node_hops": {query_key: -1},
            "edge_count": 0,
        }

    @classmethod
    def add_rewritten_query(
        cls,
        route_index: dict[str, Any],
        rewritten_query: str,
    ) -> str:
        original_key = route_index["original_query_key"]
        rewritten_node_id = cls.generate_query_id(rewritten_query)
        rewritten_key = route_node_key("query", rewritten_node_id)
        cls.record_edge(
            route_index=route_index,
            from_key=original_key,
            to_key=rewritten_key,
            from_type="query",
            from_id=route_index["query_node_id"],
            to_type="query",
            to_id=rewritten_node_id,
            relation="SAG2 query 重写",
            stage="prepare",
            method="llm_query_rewrite",
            hop=0,
            from_hop=-1,
            to_hop=-1,
            score=1.0,
        )
        route_index["query_key"] = rewritten_key
        route_index["rewritten_query_node_id"] = rewritten_node_id
        route_index["rewritten_query_text"] = rewritten_query
        return rewritten_key

    @classmethod
    def record_initial(
        cls,
        *,
        route_index: dict[str, Any],
        query_events: list[dict[str, Any]],
        query_entity_ids: list[str],
        entity_events: dict[str, list[str]],
        event_scores: dict[str, float],
    ) -> dict[str, int]:
        stats = {
            "query_event_paths": 0,
            "query_entity_paths": 0,
            "entity_event_paths": 0,
            "seed_event_key_paths": 0,
            "total_edges": 0,
            "node_count": 0,
        }
        query_key = route_index["query_key"]
        for event in query_events:
            event_id = event.get("event_id") or event.get("id")
            if not event_id:
                continue
            score = float(event.get("score") or event.get("_score") or 0.0)
            cls.record_edge(
                route_index=route_index,
                from_key=query_key,
                to_key=route_node_key("event", event_id),
                from_type="query",
                from_id=route_index["query_node_id"],
                to_type="event",
                to_id=event_id,
                relation="SAG2 query->event",
                stage="recall",
                method="query_vector_search",
                hop=0,
                from_hop=-1,
                to_hop=0,
                score=score,
            )
            stats["query_event_paths"] += 1

        for entity_id in merge_ids(query_entity_ids):
            cls.record_edge(
                route_index=route_index,
                from_key=query_key,
                to_key=route_node_key("entity", entity_id),
                from_type="query",
                from_id=route_index["query_node_id"],
                to_type="entity",
                to_id=entity_id,
                relation="SAG2 query->entity",
                stage="recall",
                method="entity_bm25",
                hop=0,
                from_hop=-1,
                to_hop=0,
                score=1.0,
            )
            stats["query_entity_paths"] += 1

        entity_route_stats = cls.record_relation(
            route_index=route_index,
            pairs_mapping=entity_events,
            event_scores=event_scores,
            method="entity_event_recall",
            relation="SAG2 entity->event",
            hop=0,
            from_hop=0,
            to_hop=0,
            key_is_event=True,
            from_is_event=False,
            stats_key="entity_event_paths",
        )
        stats["entity_event_paths"] = entity_route_stats["entity_event_paths"]
        stats["total_edges"] = route_index.get("edge_count", 0)
        stats["node_count"] = len(route_index.get("node_hops", {}))
        return stats

    @classmethod
    def record_relation(
        cls,
        *,
        route_index: dict[str, Any],
        pairs_mapping: dict[str, list[str]],
        event_scores: dict[str, float],
        method: str,
        relation: str,
        hop: int,
        from_hop: int,
        to_hop: int,
        key_is_event: bool,
        from_is_event: bool,
        stats_key: str,
    ) -> dict[str, int]:
        before_edge_count = route_index.get("edge_count", 0)
        count = 0
        for key_id, value_ids in pairs_mapping.items():
            if not key_id:
                continue
            for value_id in merge_ids(value_ids):
                if not value_id:
                    continue
                event_id = key_id if key_is_event else value_id
                entity_id = value_id if key_is_event else key_id
                if from_is_event:
                    from_type, from_id = "event", event_id
                    to_type, to_id = "entity", entity_id
                else:
                    from_type, from_id = "entity", entity_id
                    to_type, to_id = "event", event_id
                cls.record_edge(
                    route_index=route_index,
                    from_key=route_node_key(from_type, from_id),
                    to_key=route_node_key(to_type, to_id),
                    from_type=from_type,
                    from_id=from_id,
                    to_type=to_type,
                    to_id=to_id,
                    relation=relation,
                    stage="expand" if method.startswith("expand_") else "recall",
                    method=method,
                    hop=hop,
                    from_hop=from_hop,
                    to_hop=to_hop,
                    score=event_scores.get(event_id, 0.0),
                )
                count += 1
        return {
            stats_key: count,
            "total_edges": route_index.get("edge_count", 0) - before_edge_count,
        }

    @staticmethod
    def record_edge(
        *,
        route_index: dict[str, Any],
        from_key: str,
        to_key: str,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relation: str,
        stage: str,
        method: str,
        hop: int,
        from_hop: int,
        to_hop: int,
        score: float,
    ) -> None:
        incoming = route_index.setdefault("incoming", {})
        edges = incoming.setdefault(to_key, [])
        for edge in edges:
            if (
                edge.get("from") == from_key
                and edge.get("to") == to_key
                and edge.get("method") == method
                and edge.get("hop") == hop
            ):
                return
        edges.append(
            {
                "from": from_key,
                "to": to_key,
                "from_type": from_type,
                "from_id": from_id,
                "to_type": to_type,
                "to_id": to_id,
                "relation": relation,
                "stage": stage,
                "method": method,
                "hop": hop,
                "from_hop": from_hop,
                "to_hop": to_hop,
                "score": float(score or 0.0),
            }
        )
        route_index["edge_count"] = route_index.get("edge_count", 0) + 1
        route_index.setdefault("node_hops", {}).setdefault(from_key, from_hop)
        route_index.setdefault("node_hops", {}).setdefault(to_key, to_hop)
        if to_type == "event":
            route_index.setdefault("event_scores", {})[to_id] = float(score or 0.0)


__all__ = ["SAG2RouteTracker"]
