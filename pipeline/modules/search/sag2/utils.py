"""Pure helpers shared by SAG2 stages and orchestration."""

from __future__ import annotations

from typing import Any


def merge_ids(*id_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for id_group in id_groups:
        for item_id in id_group:
            if item_id and item_id not in seen:
                seen.add(item_id)
                merged.append(item_id)
    return merged


def merge_event_scores(*score_maps: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for score_map in score_maps:
        for event_id, score in score_map.items():
            if event_id not in merged or score > merged[event_id]:
                merged[event_id] = score
    return merged


def top_event_ids_by_score(event_scores: dict[str, float], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return [
        event_id
        for event_id, _ in sorted(event_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def score_range(scores: list[float]) -> str:
    if not scores:
        return "N/A"
    return f"[{min(scores):.4f}, {max(scores):.4f}]"


def round_seconds(value: float) -> float:
    return round(float(value or 0.0), 4)


def event_ids_from_events(events: list[dict[str, Any]]) -> list[str]:
    return merge_ids(
        [
            event.get("event_id") or event.get("id")
            for event in events
            if event.get("event_id") or event.get("id")
        ]
    )


def scores_from_events(events: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for event in events:
        event_id = event.get("event_id") or event.get("id")
        if not event_id:
            continue
        score = event.get("score")
        if score is None:
            score = event.get("_score")
        if score is not None:
            scores[event_id] = float(score)
    return scores


def merge_event_entity_mapping(
    target: dict[str, list[str]], source: dict[str, list[str]]
) -> None:
    for event_id, entity_ids in source.items():
        bucket = target.setdefault(event_id, [])
        for entity_id in entity_ids:
            if entity_id not in bucket:
                bucket.append(entity_id)


def build_ranked_event_results(
    event_ids: list[str],
    event_map: dict[str, dict[str, Any]],
    event_scores: dict[str, float],
    rank_scores: dict[str, float],
    rank_score_field: str,
) -> list[dict[str, Any]]:
    # ``event_map`` remains part of the stable signature for callers that pass
    # it while the result intentionally contains only ranking fields.
    del event_map
    results: list[dict[str, Any]] = []
    for event_id in event_ids:
        item = {"event_id": event_id, "score": event_scores.get(event_id, 0.0)}
        if event_id in rank_scores:
            item[rank_score_field] = rank_scores[event_id]
        results.append(item)
    return results


def event_text_for_rerank(event: dict[str, Any], event_id: str) -> str:
    parts = [
        str(event.get("title") or "").strip(),
        str(event.get("content") or "").strip(),
    ]
    text = "\n".join(part for part in parts if part)
    return text or event_id


def entity_ids_from_route_edges(edges: list[dict[str, Any]]) -> list[str]:
    entity_ids: list[str] = []
    for edge in edges:
        if edge.get("from_type") == "entity":
            entity_ids.append(edge.get("from_id", ""))
        if edge.get("to_type") == "entity":
            entity_ids.append(edge.get("to_id", ""))
    return merge_ids(entity_ids)


def event_ids_from_route_edges(edges: list[dict[str, Any]]) -> list[str]:
    event_ids: list[str] = []
    for edge in edges:
        if edge.get("from_type") == "event":
            event_ids.append(edge.get("from_id", ""))
        if edge.get("to_type") == "event":
            event_ids.append(edge.get("to_id", ""))
    return merge_ids(event_ids)


def route_node_key(node_type: str, node_id: str) -> str:
    return f"{node_type}:{node_id}"


def full_event_node(event_id: str, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event_id,
        "event_id": event_id,
        "type": "event",
        "category": event.get("category", ""),
        "title": event.get("title", ""),
        "summary": event.get("summary", ""),
        "content": event.get("content", ""),
        "stage": "recall",
    }


def full_entity_node(entity_id: str, entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entity_id,
        "entity_id": entity_id,
        "type": "entity",
        "category": entity.get("type") or "unknown",
        "name": entity.get("name") or "",
        "description": entity.get("description") or "",
    }


__all__ = [
    "merge_ids",
    "merge_event_scores",
    "top_event_ids_by_score",
    "score_range",
    "round_seconds",
    "event_ids_from_events",
    "scores_from_events",
    "merge_event_entity_mapping",
    "build_ranked_event_results",
    "event_text_for_rerank",
    "entity_ids_from_route_edges",
    "event_ids_from_route_edges",
    "route_node_key",
    "full_event_node",
    "full_entity_node",
]
