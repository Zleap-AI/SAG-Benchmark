"""Bounded SAG2 candidate universe.

The scope is deliberately independent from the query-scope route. It is built
once per request from the event vector index plus one relational bootstrap of
the event/entity edges. All subsequent SAG2 graph operations use the in-memory
maps; storage access is provided through EventUniverseStore.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SAG2CandidateSubgraph:
    """Immutable-by-convention event/entity maps for one search request."""

    event_scores: dict[str, float] = field(default_factory=dict)
    event_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    event_to_entities: dict[str, list[str]] = field(default_factory=dict)
    entity_to_events: dict[str, list[str]] = field(default_factory=dict)
    source_config_ids: tuple[str, ...] = ()

    @property
    def event_ids(self) -> set[str]:
        return set(self.event_scores)

    @property
    def entity_ids(self) -> set[str]:
        return set(self.entity_to_events)

    def contains_event(self, event_id: str | None) -> bool:
        return bool(event_id and event_id in self.event_scores)

    def contains_entity(self, entity_id: str | None) -> bool:
        return bool(entity_id and entity_id in self.entity_to_events)

    def top_events(self, limit: int) -> list[dict[str, Any]]:
        ids = sorted(self.event_scores, key=self.event_scores.get, reverse=True)[:limit]
        return [
            {
                "event_id": event_id,
                "entity_ids": list(self.event_to_entities.get(event_id, [])),
                "score": self.event_scores[event_id],
            }
            for event_id in ids
        ]

    def events_for_entities(
        self, entity_ids: list[str], *, limit_per_entity: int
    ) -> tuple[dict[str, list[str]], list[str]]:
        mapping: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for entity_id in entity_ids:
            for event_id in self.entity_to_events.get(entity_id, []):
                if counts.get(entity_id, 0) >= limit_per_entity:
                    break
                bucket = mapping.setdefault(event_id, [])
                if entity_id not in bucket:
                    bucket.append(entity_id)
                    counts[entity_id] = counts.get(entity_id, 0) + 1
        return mapping, list(mapping)

    def entities_for_events(self, event_ids: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for event_id in event_ids:
            for entity_id in self.event_to_entities.get(event_id, []):
                if entity_id not in seen:
                    seen.add(entity_id)
                    result.append(entity_id)
        return result

    def stats(self) -> dict[str, int]:
        return {
            "events": len(self.event_scores),
            "entities": len(self.entity_to_events),
            "edges": sum(len(v) for v in self.event_to_entities.values()),
        }


class SAG2CandidatePoolBuilder:
    """Build a request-scoped SAG2 subgraph with one vector and one graph-read phase."""

    def __init__(self, event_search, graph_reader):
        self._event_search = event_search
        self._graph_reader = graph_reader

    async def build(
        self,
        query_vector: list[float],
        source_config_ids: list[str],
        *,
        event_top_k: int = 1000,
        bootstrap_entity_limit: int = 0,
        include_event_content: bool = True,
    ) -> SAG2CandidateSubgraph:
        raw_events = await self._event_search(
            query_vector=query_vector,
            k=event_top_k,
            source_config_ids=source_config_ids,
        )
        details: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for raw in raw_events:
            event_id = raw.get("event_id") or raw.get("id")
            if not event_id or event_id in scores:
                continue
            score = float(raw.get("_score") or raw.get("score") or 0.0)
            scores[str(event_id)] = score
            doc = dict(raw)
            if not include_event_content:
                doc.pop("content", None)
            details[str(event_id)] = doc

        if not scores:
            return SAG2CandidateSubgraph(source_config_ids=tuple(source_config_ids))

        event_ids = list(scores)
        event_to_entities: dict[str, list[str]] = {event_id: [] for event_id in event_ids}
        entity_to_events: dict[str, list[str]] = {}
        valid_ids = set(
            await self._graph_reader.filter_active_event_ids(
                event_ids,
                source_config_ids=source_config_ids,
            )
        )
        scores = {event_id: score for event_id, score in scores.items() if event_id in valid_ids}
        details = {event_id: doc for event_id, doc in details.items() if event_id in valid_ids}
        event_ids = list(scores)
        event_to_entities = {event_id: [] for event_id in event_ids}
        if not event_ids:
            return SAG2CandidateSubgraph(source_config_ids=tuple(source_config_ids))

        relation_rows = await self._graph_reader.get_event_entity_pairs_by_events(
            event_ids,
            source_config_ids=source_config_ids,
            limit=bootstrap_entity_limit or None,
        )
        for event_id, entity_id in relation_rows:
            if event_id not in scores or not entity_id:
                continue
            if entity_id not in event_to_entities[event_id]:
                event_to_entities[event_id].append(entity_id)
            entity_to_events.setdefault(entity_id, []).append(event_id)

        return SAG2CandidateSubgraph(
            event_scores=scores,
            event_details=details,
            event_to_entities=event_to_entities,
            entity_to_events=entity_to_events,
            source_config_ids=tuple(source_config_ids),
        )
