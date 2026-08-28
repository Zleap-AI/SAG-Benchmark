"""SAG2 graph expansion stage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pipeline.modules.search.config import SAGConfig
from pipeline.utils import get_logger

from .candidate_scope import SAG2CandidateSubgraph
from .contracts import (
    SAG2ExpandResult,
    SAG2RecallResult,
    SAG2Request,
    SAG2SearchState,
)
from .evidence import EvidenceTracker
from .routes import SAG2RouteTracker
from .runtime import SAG2Runtime
from .timing import SAG2TimingService
from .utils import (
    merge_event_entity_mapping,
    merge_event_scores,
    merge_ids,
)

logger = get_logger("search.sag2.expand")


@dataclass(slots=True)
class SAG2ExpandStage:
    runtime: SAG2Runtime
    timing: SAG2TimingService

    async def run(
        self,
        request: SAG2Request,
        recall: SAG2RecallResult,
        state: SAG2SearchState,
        route_index: dict[str, Any],
        evidence: EvidenceTracker,
        timings: dict[str, float],
        retry_wasted_by_stage: dict[str, float],
    ) -> SAG2ExpandResult:
        event_ids = list(recall.event_ids)
        event_scores = dict(recall.event_scores)
        if not request.config.sag2_expand.enabled or request.config.sag2_expand.max_hops <= 0:
            return SAG2ExpandResult(
                event_ids=event_ids,
                event_scores=event_scores,
                expand_event_ids=[],
                expand_event_scores={},
                expand_entity_events={},
                seed_new_entity_events={},
                new_entity_ids=[],
                expand_new_entity_ids=[],
                expand_hops=[],
                route_stats={"seed_event_key_paths": 0},
            )

        started = time.perf_counter()
        seed_events = recall.query_events + recall.entity_event_docs
        if recall.scope is not None:
            (
                seed_new_entity_events,
                new_entity_ids,
            ) = await self._new_entities_from_seed_events_in_scope(
                seed_events,
                recall.scope,
                recall.query_vector,
                request.source_config_ids,
                request.config,
                state,
            )
        else:
            seed_new_entity_events, new_entity_ids = await self._new_entities_from_seed_events(
                events=seed_events,
                source_config_ids=request.source_config_ids,
                query_vector=recall.query_vector,
                config=request.config,
                state=state,
            )
        route_stats = SAG2RouteTracker.record_relation(
            route_index=route_index,
            pairs_mapping=seed_new_entity_events,
            event_scores=event_scores,
            method="seed_event_to_key",
            relation="SAG2 event->key",
            hop=1,
            from_hop=0,
            to_hop=1,
            key_is_event=False,
            from_is_event=True,
            stats_key="seed_event_key_paths",
        )
        state.entity_ids.update(new_entity_ids)
        seed_event_ids = [
            event.get("event_id") or event.get("id")
            for event in seed_events
            if event.get("event_id") or event.get("id")
        ]
        evidence.check("3a_expand_seed_events", merge_ids(seed_event_ids))

        if recall.scope is not None:
            expand_result = await self._expand_from_scope(
                seed_entity_ids=new_entity_ids,
                scope=recall.scope,
                config=request.config,
                state=state,
                route_index=route_index,
            )
        else:
            expand_result = await self._expand_from_entities(
                seed_entity_ids=new_entity_ids,
                query=request.query,
                source_config_ids=request.source_config_ids,
                query_vector=recall.query_vector,
                config=request.config,
                state=state,
                route_index=route_index,
            )
        (
            expand_entity_events,
            expand_event_ids,
            expand_event_scores,
            expand_new_entity_ids,
            expand_hops,
        ) = expand_result
        for hop_stat in expand_hops:
            event_ids = merge_ids(event_ids, hop_stat["event_ids"])
            evidence.check(f"4_expand_h{hop_stat['hop']}", event_ids)
        event_scores = merge_event_scores(event_scores, expand_event_scores)
        self.timing.record_timed_stage(timings, retry_wasted_by_stage, "expand", started)
        return SAG2ExpandResult(
            event_ids=event_ids,
            event_scores=event_scores,
            expand_event_ids=expand_event_ids,
            expand_event_scores=expand_event_scores,
            expand_entity_events=expand_entity_events,
            seed_new_entity_events=seed_new_entity_events,
            new_entity_ids=new_entity_ids,
            expand_new_entity_ids=expand_new_entity_ids,
            expand_hops=expand_hops,
            route_stats=route_stats,
        )

    async def _new_entities_from_seed_events_in_scope(
        self,
        events: list[dict[str, Any]],
        scope: SAG2CandidateSubgraph,
        query_vector: list[float],
        source_config_ids: list[str],
        config: SAGConfig,
        state: SAG2SearchState,
    ) -> tuple[dict[str, list[str]], list[str]]:
        ranked = sorted(events, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        seed_ids = merge_ids(
            item.get("event_id") or item.get("id")
            for item in ranked[: config.sag2_expand.seed_event_limit]
        )
        # Keep SAG2's original event_entity_vectors similarity filter, but
        # restrict its result to the pre-built in-memory universe.
        relation_map, relation_entities = await self._new_entities_from_event_ids(
            event_ids=seed_ids,
            source_config_ids=source_config_ids,
            max_entities=config.sag2_expand.entities_per_hop,
            exclude_entity_ids=state.entity_ids,
            query_vector=query_vector,
            config=config,
        )
        scoped_entities: list[str] = []
        scoped_map: dict[str, list[str]] = {}
        for entity_id in relation_entities:
            if entity_id not in scope.entity_ids:
                continue
            event_ids = [
                event_id
                for event_id in relation_map.get(entity_id, [])
                if event_id in scope.event_ids
            ]
            if not event_ids:
                continue
            scoped_entities.append(entity_id)
            scoped_map[entity_id] = event_ids
            if len(scoped_entities) >= config.sag2_expand.entities_per_hop:
                break
        return scoped_map, scoped_entities

    async def _expand_from_scope(
        self,
        seed_entity_ids: list[str],
        scope: SAG2CandidateSubgraph,
        config: SAGConfig,
        state: SAG2SearchState,
        route_index: dict[str, Any],
    ) -> tuple[
        dict[str, list[str]],
        list[str],
        dict[str, float],
        list[str],
        list[dict[str, Any]],
    ]:
        current_entities = merge_ids(seed_entity_ids)
        expand_mapping: dict[str, list[str]] = {}
        expand_ids: list[str] = []
        expand_scores: dict[str, float] = {}
        new_entities_all: list[str] = []
        hops: list[dict[str, Any]] = []
        for hop_no in range(1, config.sag2_expand.max_hops + 1):
            if not current_entities:
                break
            candidate_map, candidate_ids = scope.events_for_entities(
                current_entities, limit_per_entity=config.sag2_expand.max_events_per_hop
            )
            candidate_ids = [eid for eid in candidate_ids if eid not in state.seen_event_ids]
            candidate_ids = sorted(
                candidate_ids,
                key=lambda eid: scope.event_scores.get(eid, 0.0),
                reverse=True,
            )[: config.sag2_expand.max_events_per_hop]
            hop_scores = {eid: scope.event_scores[eid] for eid in candidate_ids}
            threshold = config.sag2_expand.event_similarity_threshold
            candidate_ids = [eid for eid in candidate_ids if hop_scores[eid] >= threshold]
            hop_scores = {eid: hop_scores[eid] for eid in candidate_ids}
            candidate_map = {eid: candidate_map.get(eid, []) for eid in candidate_ids}
            merge_event_entity_mapping(expand_mapping, candidate_map)
            expand_ids = merge_ids(expand_ids, candidate_ids)
            expand_scores = merge_event_scores(expand_scores, hop_scores)
            state.seen_event_ids.update(candidate_ids)
            SAG2RouteTracker.record_relation(
                route_index=route_index,
                pairs_mapping=candidate_map,
                event_scores=hop_scores,
                method="expand_key_to_event",
                relation="SAG2 scoped entity->event",
                hop=hop_no,
                from_hop=hop_no,
                to_hop=hop_no,
                key_is_event=True,
                from_is_event=False,
                stats_key="entity_event_paths",
            )
            next_entities = [
                entity_id
                for entity_id in scope.entities_for_events(candidate_ids)
                if entity_id not in state.entity_ids
            ][: config.sag2_expand.entities_per_hop]
            state.entity_ids.update(next_entities)
            new_entities_all = merge_ids(new_entities_all, next_entities)
            hops.append(
                {
                    "hop": hop_no,
                    "is_last_hop": hop_no == config.sag2_expand.max_hops,
                    "key_count": len(current_entities),
                    "event_count": len(candidate_ids),
                    "new_entity_count": len(next_entities),
                    "event_ids": candidate_ids,
                }
            )
            current_entities = next_entities
        return expand_mapping, expand_ids, expand_scores, new_entities_all, hops

    async def _new_entities_from_seed_events(
        self,
        events: list[dict[str, Any]],
        source_config_ids: list[str],
        query_vector: list[float],
        config: SAGConfig,
        state: SAG2SearchState,
    ) -> tuple[dict[str, list[str]], list[str]]:
        """从种子事项出发，用 ES event_entity_vectors 关系向量相似度选新实体（排除已用）。

        种子事项先按 query 相似度取 top-seed_event_limit 限流，
        再调 _new_entities_from_event_ids 做 ES kNN 关系向量打分。
        """
        if not events:
            return {}, []

        seed_limit = config.sag2_expand.seed_event_limit
        ranked = sorted(events, key=lambda e: float(e.get("score") or 0.0), reverse=True)
        seed_event_ids: list[str] = []
        seen: set[str] = set()
        for ev in ranked:
            if seed_limit > 0 and len(seed_event_ids) >= seed_limit:
                break
            event_id = ev.get("event_id") or ev.get("id")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            seed_event_ids.append(event_id)

        if not seed_event_ids:
            return {}, []

        entity_to_event_ids, new_entity_ids = await self._new_entities_from_event_ids(
            event_ids=seed_event_ids,
            source_config_ids=source_config_ids,
            max_entities=config.sag2_expand.entities_per_hop,
            exclude_entity_ids=state.entity_ids,
            query_vector=query_vector,
            config=config,
        )
        logger.info(
            "[SAG2] seed events -> new entities: seeds=%d, new_entities=%d",
            len(seed_event_ids),
            len(new_entity_ids),
        )
        return entity_to_event_ids, new_entity_ids

    async def _new_entities_from_event_ids(
        self,
        event_ids: list[str],
        source_config_ids: list[str],
        max_entities: int,
        exclude_entity_ids: set[str],
        query_vector: list[float],
        config: SAGConfig,
    ) -> tuple[dict[str, list[str]], list[str]]:
        """event_ids -> new entity_ids（ES kNN 关系向量相似度打分，排除 exclude）。

        用 event_entity_vectors 索引的 search_similar_by_description 做 kNN，
        relation_k = max_entities × relation_k_multiplier，过滤 entity_relation_score_threshold。
        对 (event, entity) 关系按 query 向量相似度评分并筛选新实体。
        """
        event_ids = merge_ids(event_ids)
        if not event_ids or max_entities <= 0:
            return {}, []

        relation_k = max_entities * config.sag2_expand.relation_k_multiplier
        threshold = config.sag2_expand.entity_relation_score_threshold

        logger.info(
            "[SAG2] event -> new key 关系向量召回: events=%d, relation_k=%d, threshold=%.4f",
            len(event_ids),
            relation_k,
            threshold,
        )

        # ES kNN on event_entity_vectors，用 event_ids 过滤 + query_vector 打分
        relations = await self.runtime.search_event_entities_by_vector(
            query_vector=query_vector,
            k=relation_k,
            event_ids=event_ids,
            source_config_ids=source_config_ids,
        )

        entity_to_event_ids: dict[str, list[str]] = {}
        new_entity_ids: list[str] = []
        seen_entities: set[str] = set()
        raw_count = 0
        kept_count = 0

        for rel in relations:
            raw_count += 1
            event_id = rel.get("event_id")
            entity_id = rel.get("entity_id")
            if not event_id or not entity_id:
                continue
            if entity_id in exclude_entity_ids:
                continue

            score = float(rel.get("_score") or rel.get("score") or 0.0)
            if score < threshold:
                continue
            kept_count += 1

            if entity_id not in seen_entities:
                if len(new_entity_ids) >= max_entities:
                    continue
                seen_entities.add(entity_id)
                new_entity_ids.append(entity_id)
                entity_to_event_ids[entity_id] = []

            related = entity_to_event_ids[entity_id]
            if event_id not in related:
                related.append(event_id)

        logger.info(
            "[SAG2] event -> new key 关系向量召回完成: raw=%d, kept=%d, entities=%d, top=%s",
            raw_count,
            kept_count,
            len(new_entity_ids),
            new_entity_ids[:5],
        )
        return entity_to_event_ids, new_entity_ids

    async def _events_from_entities(
        self,
        entity_ids: list[str],
        source_config_ids: list[str],
        *,
        max_events_per_key: int,
        exclude_event_ids: set[str],
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Resolve expansion entities through the runtime storage port."""
        return await self.runtime.retrieve_entity_event_pairs(
            entity_ids,
            source_config_ids,
            max_events_per_entity=max_events_per_key,
            exclude_event_ids=exclude_event_ids,
        )

    async def _events_from_new_entities(
        self,
        entity_ids: list[str],
        query: str,
        source_config_ids: list[str],
        query_vector: list[float],
        config: SAGConfig,
        state: SAG2SearchState,
        max_events: int,
    ) -> tuple[dict[str, list[str]], list[str], dict[str, float]]:
        """new entity -> new event：MySQL 取候选（排除 seen）+ coarse rank 过滤。"""
        entity_ids = merge_ids(entity_ids)
        if not entity_ids or max_events <= 0:
            return {}, [], {}

        # MySQL: entity -> event（排除已见），每实体不设上限（受 coarse rank 截断）
        candidate_entity_events, candidate_event_ids = await self._events_from_entities(
            entity_ids=entity_ids,
            source_config_ids=source_config_ids,
            max_events_per_key=max_events,
            exclude_event_ids=state.seen_event_ids,
        )
        if not candidate_event_ids:
            return {}, [], {}

        # coarse rank 相似度打分 + event_similarity_threshold 过滤
        ranked = await self.runtime.coarse_rank_events(
            event_ids=candidate_event_ids,
            source_config_ids=source_config_ids,
            max_events=max_events,
            query_vector=query_vector,
        )
        threshold = config.sag2_expand.event_similarity_threshold
        event_to_entity_ids: dict[str, list[str]] = {}
        event_ids: list[str] = []
        event_scores: dict[str, float] = {}
        for item in ranked:
            event_id = item.get("event_id")
            if not event_id or event_id not in candidate_entity_events:
                continue
            score = float(item.get("score") or 0.0)
            if score < threshold:
                continue
            event_ids.append(event_id)
            event_to_entity_ids[event_id] = candidate_entity_events[event_id]
            event_scores[event_id] = score

        logger.info(
            "[SAG2] new entity -> new event: candidates=%d, kept=%d",
            len(candidate_event_ids),
            len(event_ids),
        )
        return event_to_entity_ids, event_ids, event_scores

    async def _expand_from_entities(
        self,
        seed_entity_ids: list[str],
        query: str,
        source_config_ids: list[str],
        query_vector: list[float],
        config: SAGConfig,
        state: SAG2SearchState,
        route_index: dict[str, Any],
    ) -> tuple[
        dict[str, list[str]],
        list[str],
        dict[str, float],
        list[str],
        list[dict[str, Any]],
    ]:
        """for 循环多跳：new key -> new event，非末跳再 event -> new key。"""
        current_entity_ids = merge_ids(seed_entity_ids)
        expand_entity_events: dict[str, list[str]] = {}
        expand_event_ids: list[str] = []
        expand_event_scores: dict[str, float] = {}
        expand_new_entity_ids: list[str] = []
        hop_stats: list[dict[str, Any]] = []

        hop_count = config.sag2_expand.max_hops
        for hop in range(hop_count):
            if not current_entity_ids:
                logger.info("[SAG2] expand 第 %d 跳无新实体，停止", hop + 1)
                break
            is_last_hop = hop == hop_count - 1
            hop_no = hop + 1

            (
                hop_entity_events,
                hop_event_ids,
                hop_event_scores,
            ) = await self._events_from_new_entities(
                entity_ids=current_entity_ids,
                query=query,
                source_config_ids=source_config_ids,
                query_vector=query_vector,
                config=config,
                state=state,
                max_events=config.sag2_expand.max_events_per_hop,
            )

            SAG2RouteTracker.record_relation(
                route_index=route_index,
                pairs_mapping=hop_entity_events,
                event_scores=hop_event_scores,
                method="expand_key_to_event",
                relation="SAG2 expand key->event",
                hop=hop_no,
                from_hop=hop_no,
                to_hop=hop_no,
                key_is_event=True,
                from_is_event=False,
                stats_key="entity_event_paths",
            )
            merge_event_entity_mapping(expand_entity_events, hop_entity_events)
            expand_event_ids = merge_ids(expand_event_ids, hop_event_ids)
            expand_event_scores = merge_event_scores(expand_event_scores, hop_event_scores)
            state.seen_event_ids.update(hop_event_ids)

            hop_new_entity_events: dict[str, list[str]] = {}
            hop_new_entity_ids: list[str] = []
            if not is_last_hop:
                hop_new_entity_events, hop_new_entity_ids = await self._new_entities_from_event_ids(
                    event_ids=hop_event_ids,
                    source_config_ids=source_config_ids,
                    max_entities=config.sag2_expand.entities_per_hop,
                    exclude_entity_ids=state.entity_ids,
                    query_vector=query_vector,
                    config=config,
                )
                SAG2RouteTracker.record_relation(
                    route_index=route_index,
                    pairs_mapping=hop_new_entity_events,
                    event_scores=hop_event_scores,
                    method="expand_event_to_key",
                    relation="SAG2 expand event->key",
                    hop=hop_no,
                    from_hop=hop_no,
                    to_hop=hop_no + 1,
                    key_is_event=False,
                    from_is_event=True,
                    stats_key="event_key_paths",
                )
                expand_new_entity_ids = merge_ids(expand_new_entity_ids, hop_new_entity_ids)
                state.entity_ids.update(hop_new_entity_ids)

            hop_stats.append(
                {
                    "hop": hop_no,
                    "is_last_hop": is_last_hop,
                    "key_count": len(current_entity_ids),
                    "event_count": len(hop_event_ids),
                    "new_entity_count": len(hop_new_entity_ids),
                    "event_ids": hop_event_ids,
                }
            )
            logger.info(
                "[SAG2] expand 第 %d 跳完成: events=%d, new_keys=%d",
                hop_no,
                len(hop_event_ids),
                len(hop_new_entity_ids),
            )
            current_entity_ids = hop_new_entity_ids

        return (
            expand_entity_events,
            expand_event_ids,
            expand_event_scores,
            expand_new_entity_ids,
            hop_stats,
        )


__all__ = ["SAG2ExpandStage"]
