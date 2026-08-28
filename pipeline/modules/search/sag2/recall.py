"""SAG2 recall algorithm and stage orchestration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pipeline.modules.search.config import SAGConfig
from pipeline.utils import get_logger

from .candidate_scope import SAG2CandidatePoolBuilder, SAG2CandidateSubgraph
from .contracts import (
    SAG2Entity,
    SAG2RecallResult,
    SAG2Request,
    SAG2RewriteOutput,
    SAG2SearchState,
)
from .evidence import EvidenceTracker
from .routes import SAG2RouteTracker
from .runtime import SAG2Runtime
from .timing import SAG2TimingService
from .utils import (
    event_ids_from_events,
    merge_event_scores,
    merge_ids,
    score_range,
    scores_from_events,
)

logger = get_logger("search.sag2.recall")


@dataclass(slots=True)
class SAG2RecallStage:
    runtime: SAG2Runtime
    timing: SAG2TimingService

    async def run(
        self,
        request: SAG2Request,
        state: SAG2SearchState,
        route_index: dict[str, Any],
        evidence: EvidenceTracker,
        timings: dict[str, float],
        retry_wasted_by_stage: dict[str, float],
    ) -> SAG2RecallResult:
        config = request.config
        query = request.query
        source_config_ids = request.source_config_ids

        t0 = time.perf_counter()
        retry_before = self.timing.retry_wasted_snapshot()
        rewritten_query, rewritten_entities = await self._rewrite_query_and_extract_entities(
            query, config
        )
        effective_query = rewritten_query or query
        if rewritten_query:
            SAG2RouteTracker.add_rewritten_query(route_index, rewritten_query)
        self.timing.record_timed_stage(
            timings,
            retry_wasted_by_stage,
            "rewrite_query",
            t0,
            retry_before,
            self.timing.retry_wasted_snapshot(),
        )

        t0 = time.perf_counter()
        processor = await self.runtime.get_processor()
        query_vector = await processor.generate_embedding(effective_query)
        self.timing.record_timed_stage(timings, retry_wasted_by_stage, "query_embedding", t0)

        t0 = time.perf_counter()
        scope = None
        if config.sag2_scope.enabled:
            scope = await SAG2CandidatePoolBuilder(
                self.runtime.search_events_by_vector,
                self.runtime.event_universe_store,
            ).build(
                query_vector=query_vector,
                source_config_ids=source_config_ids,
                event_top_k=config.sag2_scope.event_top_k,
                bootstrap_entity_limit=config.sag2_scope.bootstrap_entity_limit,
                include_event_content=config.sag2_scope.include_event_content,
            )
        if scope is not None:
            self.timing.record_timed_stage(timings, retry_wasted_by_stage, "candidate_pool", t0)
            logger.info("[SAG2] candidate scope enabled: %s", scope.stats())

        t0 = time.perf_counter()
        retry_before = self.timing.retry_wasted_snapshot()
        if scope is not None:
            query_result, entity_result = await asyncio.gather(
                self._recall_query_to_event_in_scope(scope, config),
                self._recall_entity_event_candidates_in_scope(
                    query,
                    effective_query,
                    rewritten_entities,
                    source_config_ids,
                    config,
                    state,
                    scope,
                ),
            )
        else:
            query_result, entity_result = await asyncio.gather(
                self._recall_query_to_event(query_vector, source_config_ids, config),
                self._recall_entity_event_candidates(
                    query,
                    effective_query,
                    rewritten_entities,
                    source_config_ids,
                    config,
                    state,
                ),
            )
        query_events, event_score_cache = query_result
        entity_events, entity_event_ids_pre, query_entity_ids = entity_result
        self.timing.record_timed_stage(
            timings,
            retry_wasted_by_stage,
            "parallel_recall",
            t0,
            retry_before,
            self.timing.retry_wasted_snapshot(),
        )
        evidence.check("1_path_a_query_event", event_ids_from_events(query_events))
        evidence.check("1_path_b_entity_event_raw", entity_event_ids_pre)

        t0 = time.perf_counter()
        if scope is not None:
            (
                entity_events,
                entity_event_ids,
                entity_event_docs,
            ) = await self._filter_entity_events_by_similarity_in_scope(
                entity_event_ids_pre, entity_events, scope, config
            )
        else:
            (
                entity_events,
                entity_event_ids,
                entity_event_docs,
            ) = await self._filter_entity_events_by_similarity(
                event_ids=entity_event_ids_pre,
                entity_events=entity_events,
                query_vector=query_vector,
                source_config_ids=source_config_ids,
                config=config,
                score_cache=event_score_cache,
            )
        self.timing.record_timed_stage(
            timings, retry_wasted_by_stage, "event_similarity_filter", t0
        )
        evidence.check("2_path_b_similarity_filter", entity_event_ids)

        t0 = time.perf_counter()
        query_event_ids = event_ids_from_events(query_events)
        query_event_scores = scores_from_events(query_events)
        entity_event_scores = scores_from_events(entity_event_docs)
        event_ids = merge_ids(query_event_ids, entity_event_ids)
        event_scores = merge_event_scores(query_event_scores, entity_event_scores)
        state.seen_event_ids.update(event_ids)
        logger.info(
            "[SAG2] events merged: query=%d, entity=%d, merged=%d, score_range=%s",
            len(query_event_ids),
            len(entity_event_ids),
            len(event_ids),
            score_range([event_scores[e] for e in event_ids if e in event_scores]),
        )
        evidence.check("3_recall_merged", event_ids)
        initial_route_stats = SAG2RouteTracker.record_initial(
            route_index=route_index,
            query_events=query_events,
            query_entity_ids=query_entity_ids,
            entity_events=entity_events,
            event_scores=event_scores,
        )
        self.timing.record_timed_stage(timings, retry_wasted_by_stage, "merge_routes", t0)

        return SAG2RecallResult(
            rewritten_query=rewritten_query,
            rewritten_entities=rewritten_entities,
            effective_query=effective_query,
            query_vector=query_vector,
            scope=scope,
            query_events=query_events,
            entity_events=entity_events,
            entity_event_docs=entity_event_docs,
            query_event_ids=query_event_ids,
            entity_event_ids=entity_event_ids,
            query_entity_ids=query_entity_ids,
            event_ids=event_ids,
            event_scores=event_scores,
            initial_route_stats=initial_route_stats,
        )

    async def _retrieve_entity_candidates(
        self,
        query_entities: list[str],
        source_config_ids: list[str],
        *,
        entity_top_k: int,
        key_similarity_threshold: float,
        state: SAG2SearchState,
        allowed_entity_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[float]]:
        """SAG2 wrapper around the runtime entity retrieval dependency."""

        result = await self.runtime.retrieve_entity_candidates(
            query_entities,
            source_config_ids,
            entity_top_k=entity_top_k,
            key_similarity_threshold=key_similarity_threshold,
            allowed_entity_ids=allowed_entity_ids,
        )
        state.entity_ids.update(result[0])
        logger.info(
            "[SAG2] entity vector recall: query_entities=%d, entities=%d",
            len(query_entities),
            len(result[0]),
        )
        return result

    async def _recall_query_to_event_in_scope(
        self, scope: SAG2CandidateSubgraph, config: SAGConfig
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        threshold = config.sag2_recall.score_threshold
        score_cache = {
            event_id: score for event_id, score in scope.event_scores.items() if score >= threshold
        }
        events = [
            event
            for event in scope.top_events(config.sag2_recall.query_recall_event_max)
            if event["event_id"] in score_cache
        ]
        return events, score_cache

    async def _recall_entity_event_candidates_in_scope(
        self,
        query: str,
        effective_query: str,
        rewritten_entities: list[SAG2Entity],
        source_config_ids: list[str],
        config: SAGConfig,
        state: SAG2SearchState,
        scope: SAG2CandidateSubgraph,
    ) -> tuple[dict[str, list[str]], list[str], list[str]]:
        if not scope.entity_ids:
            return {}, [], []
        if config.sag2_enable_entity_extraction:
            ner_entities = await self.runtime.extract_entities(query, config)
            self.timing.record_llm_call_metrics()
            query_entity_ids, _names, _scores = (
                await self._retrieve_entity_candidates(
                    query_entities=ner_entities,
                    source_config_ids=source_config_ids,
                    entity_top_k=config.sag2_recall.max_entities,
                    key_similarity_threshold=config.sag2_recall.entity_vector_threshold,
                    state=state,
                    allowed_entity_ids=list(scope.entity_ids),
                )
                if ner_entities
                else ([], [], [])
            )
        else:
            entities = await self.runtime.search_entities_by_text(
                query=query,
                source_config_ids=source_config_ids,
                size=config.sag2_recall.max_entities,
                allowed_entity_ids=list(scope.entity_ids),
            )
            query_entity_ids = [e.get("entity_id") for e in entities if e.get("entity_id")]
        query_entity_ids = merge_ids(
            entity_id for entity_id in query_entity_ids if entity_id in scope.entity_ids
        )
        state.entity_ids.update(query_entity_ids)
        entity_events, event_ids = scope.events_for_entities(
            query_entity_ids, limit_per_entity=config.sag2_recall.max_events_per_key
        )
        return entity_events, event_ids, query_entity_ids

    async def _filter_entity_events_by_similarity_in_scope(
        self,
        event_ids: list[str],
        entity_events: dict[str, list[str]],
        scope: SAG2CandidateSubgraph,
        config: SAGConfig,
    ) -> tuple[dict[str, list[str]], list[str], list[dict[str, Any]]]:
        threshold = config.sag2_recall.score_threshold
        kept_ids = [
            event_id
            for event_id in merge_ids(event_ids)
            if event_id in scope.event_scores and scope.event_scores[event_id] >= threshold
        ]
        filtered = {event_id: list(entity_events.get(event_id, [])) for event_id in kept_ids}
        docs = [
            {
                "event_id": event_id,
                "entity_ids": filtered[event_id],
                "score": scope.event_scores[event_id],
            }
            for event_id in kept_ids
        ]
        return filtered, kept_ids, docs

    async def _recall_query_to_event(
        self,
        query_vector: list[float],
        source_config_ids: list[str],
        config: SAGConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """路 A：query -> event 直接向量召回。

        返回 (top_events, score_cache)：score_cache 是过阈值全量 event_id -> score，
        供路 B 相似度过滤复用。entity_ids 读 ES event 文档自带字段。
        """
        k = config.sag2_recall.query_recall_event_max
        threshold = config.sag2_recall.score_threshold

        raw_events = await self.runtime.search_events_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
        )

        results: list[dict[str, Any]] = []
        score_cache: dict[str, float] = {}
        seen: set[str] = set()
        for raw in raw_events:
            event_id = raw.get("event_id") or raw.get("id")
            if not event_id or event_id in seen:
                continue
            score = float(raw.get("_score") or raw.get("score") or 0.0)
            if score < threshold:
                continue
            seen.add(event_id)
            score_cache[event_id] = score
            if len(results) >= k:
                continue
            entity_ids = raw.get("entity_ids") or []
            if not isinstance(entity_ids, list):
                entity_ids = [str(entity_ids)]
            results.append({"event_id": event_id, "entity_ids": entity_ids, "score": score})

        logger.info(
            "[SAG2] 路A query->event: raw=%d, kept=%d, cache=%d",
            len(raw_events),
            len(results),
            len(score_cache),
        )
        return results, score_cache

    async def _recall_entity_event_candidates(
        self,
        query: str,
        effective_query: str,
        rewritten_entities: list[SAG2Entity],
        source_config_ids: list[str],
        config: SAGConfig,
        state: SAG2SearchState,
    ) -> tuple[dict[str, list[str]], list[str], list[str]]:
        """路 B 前段：query -> entities -> event_ids（不打分）。

        实体入口：默认 BM25；若 sag2_enable_entity_extraction 则用 SAG2 的 step1/step2
        （NER -> 实体向量 top-k）。entity->event 走 EventUniverseStore 的规范关系读取。

        返回 (entity_events: {event_id: [entity_id]}, entity_event_ids_pre, query_entity_ids)。
        """
        query_entity_ids: list[str] = []

        if config.sag2_enable_entity_extraction:
            # NER -> 实体向量（复用 SAG2 step1/step2）
            ner_entities = await self.runtime.extract_entities(query, config)
            self.timing.record_llm_call_metrics()
            if ner_entities:
                entity_ids, _names, _scores = await self._retrieve_entity_candidates(
                    query_entities=ner_entities,
                    source_config_ids=source_config_ids,
                    entity_top_k=config.sag2_recall.max_entities,
                    key_similarity_threshold=config.sag2_recall.entity_vector_threshold,
                    state=state,
                )
                query_entity_ids = merge_ids(entity_ids)
        else:
            # 默认：BM25 实体召回
            entities = await self.runtime.search_entities_by_text(
                query=query,
                source_config_ids=source_config_ids,
                size=config.sag2_recall.max_entities,
            )
            query_entity_ids = merge_ids(
                [e.get("entity_id") for e in entities if e.get("entity_id")]
            )

        state.entity_ids.update(query_entity_ids)
        logger.info("[SAG2] 路B 实体召回: entities=%d", len(query_entity_ids))

        if not query_entity_ids:
            return {}, [], query_entity_ids

        entity_events, event_ids = await self.events_from_entities(
            entity_ids=query_entity_ids,
            source_config_ids=source_config_ids,
            config=config,
        )
        return entity_events, event_ids, query_entity_ids

    async def events_from_entities(
        self,
        entity_ids: list[str],
        source_config_ids: list[str],
        config: SAGConfig,
        *,
        max_events_per_key: int | None = None,
        exclude_event_ids: set[str] | None = None,
    ) -> tuple[dict[str, list[str]], list[str]]:
        """MySQL：entity_ids -> event_id -> entity_ids 映射（布尔关系，每实体限量）。"""
        per_key = max_events_per_key or config.sag2_recall.max_events_per_key
        event_to_entity_ids, event_ids = await self.runtime.retrieve_entity_event_pairs(
            entity_ids,
            source_config_ids,
            max_events_per_entity=per_key,
            exclude_event_ids=exclude_event_ids,
        )
        logger.info(
            "[SAG2] entity->event (MySQL): entities=%d, events=%d",
            len(merge_ids(entity_ids)),
            len(event_ids),
        )
        return event_to_entity_ids, event_ids

    async def _filter_entity_events_by_similarity(
        self,
        event_ids: list[str],
        entity_events: dict[str, list[str]],
        query_vector: list[float],
        source_config_ids: list[str],
        config: SAGConfig,
        *,
        score_cache: dict[str, float] | None = None,
    ) -> tuple[dict[str, list[str]], list[str], list[dict[str, Any]]]:
        """对 entity->event 候选做 query 相似度过滤。

        命中 score_cache 的 event 直接复用路 A 分数；未命中的用 coarse_rank_events
        （event_ids 过滤 kNN）打分，再过 score_threshold。
        """
        event_ids = merge_ids(event_ids)
        if not event_ids:
            return {}, [], []

        cache = score_cache or {}
        threshold = config.sag2_recall.score_threshold
        cached_ids = [e for e in event_ids if e in cache]
        uncached_ids = [e for e in event_ids if e not in cache]

        filtered_events: list[dict[str, Any]] = []
        filtered_event_ids: list[str] = []
        filtered_entity_events: dict[str, list[str]] = {}

        # 1) 缓存命中（分数已过阈值）
        for event_id in cached_ids:
            score = cache[event_id]
            filtered_entity_events[event_id] = entity_events[event_id]
            filtered_event_ids.append(event_id)
            filtered_events.append(
                {"event_id": event_id, "entity_ids": entity_events[event_id], "score": score}
            )

        # 2) 缓存外：Step6 打分（event_ids 过滤 kNN）
        if uncached_ids:
            ranked = await self.runtime.coarse_rank_events(
                event_ids=uncached_ids,
                source_config_ids=source_config_ids,
                max_events=len(uncached_ids),
                query_vector=query_vector,
            )
            for item in ranked:
                event_id = item.get("event_id")
                if not event_id or event_id not in entity_events:
                    continue
                score = float(item.get("score") or 0.0)
                if score < threshold:
                    continue
                filtered_entity_events[event_id] = entity_events[event_id]
                filtered_event_ids.append(event_id)
                filtered_events.append(
                    {
                        "event_id": event_id,
                        "entity_ids": entity_events[event_id],
                        "score": score,
                    }
                )

        logger.info(
            "[SAG2] entity events 相似度过滤: before=%d, cached=%d, after=%d",
            len(event_ids),
            len(cached_ids),
            len(filtered_event_ids),
        )
        return filtered_entity_events, filtered_event_ids, filtered_events

    async def _rewrite_query_and_extract_entities(
        self,
        query: str,
        config: SAGConfig,
    ) -> tuple[str | None, list[SAG2Entity]]:
        """问题重写 + 实体提取。关闭/失败时返回 (None, [])。"""
        if not config.sag2_rewrite_query_enabled:
            return None, []

        llm_client = None
        try:
            current_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            messages = self.runtime.get_prompts(config).get_sag2_rewrite_messages(
                query=query,
                current_timestamp=current_ts,
                max_entities=config.sag2_recall.max_entities,
            )
            llm_client = await self.runtime.get_llm_client()
            parsed: SAG2RewriteOutput = await llm_client.chat_parsed(messages, SAG2RewriteOutput)
            self.timing.record_llm_call_metrics()
        except Exception as exc:
            if getattr(llm_client, "_last_call_timing", None):
                self.timing.record_llm_call_metrics()
            logger.warning("[SAG2] query 重写失败，回退原始 query: %s", exc)
            return None, []

        rewritten_query = (parsed.rewritten_query or "").strip()
        if not rewritten_query:
            logger.warning("[SAG2] query 重写结果为空，回退原始 query")
            return None, []
        logger.info(
            "[SAG2] query 重写完成: original=%r, rewritten=%r, entities=%d",
            query,
            rewritten_query,
            len(parsed.entities),
        )
        return rewritten_query, parsed.entities


__all__ = ["SAG2RecallStage"]
