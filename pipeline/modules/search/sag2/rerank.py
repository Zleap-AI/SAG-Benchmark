"""SAG2 ranking and fallback stage."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from pipeline.modules.search.config import SAGConfig
from pipeline.utils import get_logger

from .contracts import SAG2Request, SAG2RerankOutput, SAG2RerankResult
from .evidence import EvidenceTracker
from .runtime import SAG2Runtime
from .timing import SAG2TimingService
from .utils import (
    build_ranked_event_results,
    event_text_for_rerank,
    merge_ids,
)

logger = get_logger("search.sag2.rerank")


@dataclass(slots=True)
class SAG2RerankStage:
    runtime: SAG2Runtime
    timing: SAG2TimingService
    rerank_client: Any = None
    settings: Any = None

    async def run(
        self,
        request: SAG2Request,
        event_ids: list[str],
        event_scores: dict[str, float],
        event_details: dict[str, dict[str, Any]] | None,
        evidence: EvidenceTracker,
        timings: dict[str, float],
        retry_wasted_by_stage: dict[str, float],
    ) -> SAG2RerankResult:
        started = time.perf_counter()
        retry_before = self.timing.retry_wasted_snapshot()
        (
            rank_events,
            rank_event_ids,
            rank_scores,
            rank_failed,
            rank_method,
            rank_event_map,
            display_event_ids,
            failure_reason,
            retry_count,
        ) = await self.rank_events(
            query=request.query,
            event_ids=event_ids,
            event_scores=event_scores,
            source_config_ids=request.source_config_ids,
            config=request.config,
            event_details=event_details,
        )
        self.timing.record_timed_stage(
            timings,
            retry_wasted_by_stage,
            "rerank",
            started,
            retry_before,
            self.timing.retry_wasted_snapshot(),
        )
        evidence.check("6_rank_final", rank_event_ids)
        return SAG2RerankResult(
            rank_events=rank_events,
            rank_event_ids=rank_event_ids,
            rank_scores=rank_scores,
            rank_failed=rank_failed,
            rank_method=rank_method,
            rank_event_map=rank_event_map,
            display_event_ids=display_event_ids,
            failure_reason=failure_reason,
            retry_count=retry_count,
        )


    async def rank_events(
        self,
        query: str,
        event_ids: list[str],
        event_scores: dict[str, float],
        source_config_ids: list[str],
        config: SAGConfig,
        event_details: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        dict[str, float],
        bool,
        str,
        dict[str, dict[str, Any]],
        list[str],
        str | None,
        int,
    ]:
        """排序阶段入口：按 config.sag2_rerank.strategy 分发。

        末尾额外返回 display_event_ids：用于前端展示的"真正精排选中"子集，
        不含 rerank / llm_rank 策略里"不足 max_results 时按相似度补齐"的那部分。
        """
        strategy = config.sag2_rerank.strategy
        if strategy == "llm_rank":
            events, ids, scores, failed, event_map, display_ids, reason, retries = (
                await self._llm_rank_events(
                    query, event_ids, event_scores, source_config_ids, config, event_details
                )
            )
            return (
                events,
                ids,
                scores,
                failed,
                "llm_rank",
                event_map,
                display_ids,
                reason,
                retries,
            )
        if strategy == "rrf":
            events, ids, scores, failed, event_map = await self._rrf_rank_events(
                query, event_ids, event_scores, source_config_ids, config, event_details
            )
            return events, ids, scores, failed, "rrf", event_map, ids, None, 0
        events, ids, scores, failed, event_map = await self.rerank_events(
            query, event_ids, event_scores, source_config_ids, config, event_details
        )
        display_ids = [event_id for event_id in ids if event_id in scores]
        return events, ids, scores, failed, "rerank", event_map, display_ids, None, 0


    async def _load_rank_candidate_events(
        self,
        event_ids: list[str],
        source_config_ids: list[str],
        event_details: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """加载排序候选事件详情（ES get_events_by_ids）。"""
        event_ids = merge_ids(event_ids)
        if not event_ids:
            return [], {}
        if event_details is not None:
            event_map = {event_id: event_details[event_id] for event_id in event_ids if event_id in event_details}
            return [event_id for event_id in event_ids if event_id in event_map], event_map
        event_docs = await self.runtime.get_events_by_ids(
            event_ids=event_ids,
            source_includes=["event_id", "title", "summary", "content", "category"],
        )
        event_map = {
            (d.get("event_id") or d.get("id")): d
            for d in event_docs
            if d.get("event_id") or d.get("id")
        }
        ordered = [e for e in event_ids if e in event_map]
        logger.info("[SAG2] 排序候选详情: input=%d, loaded=%d", len(event_ids), len(ordered))
        return ordered, event_map


    async def rerank_events(
        self,
        query: str,
        event_ids: list[str],
        event_scores: dict[str, float],
        source_config_ids: list[str],
        config: SAGConfig,
        event_details: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, float], bool, dict[str, dict[str, Any]]]:
        """rerank 模型精排（成功不足时补齐，超时整体降级）。"""
        event_ids = merge_ids(event_ids)
        if not event_ids:
            return [], [], {}, False, {}

        ordered_event_ids, event_map = await self._load_rank_candidate_events(
            event_ids, source_config_ids, event_details
        )
        if not ordered_event_ids:
            return [], [], {}, False, event_map

        documents = [
            {"id": e, "text": event_text_for_rerank(event_map[e], e)}
            for e in ordered_event_ids
        ]
        rerank_top_k = min(
            config.sag2_rerank.rerank_top_k,
            config.sag2_rerank.max_results,
            len(documents),
        )
        try:
            rerank_client = self.rerank_client or self.runtime.get_rerank_client()
            settings = self.settings or self.runtime.get_settings()
            rerank_results = await asyncio.wait_for(
                rerank_client.rerank(
                    query=query,
                    documents=documents,
                    top_n=rerank_top_k,
                    use_prompt_template=settings.server_type == "LOCAL",
                ),
                timeout=config.sag2_rerank.rerank_timeout,
            )
        except Exception as exc:
            logger.warning("[SAG2] rerank 失败/超时，降级为相似度排序: %s", exc)
            fallback_ids = ordered_event_ids[: config.sag2_rerank.max_results]
            return (
                build_ranked_event_results(
                    fallback_ids, event_map, event_scores, {}, "rerank_score"
                ),
                fallback_ids,
                {},
                True,
                event_map,
            )

        rerank_scores: dict[str, float] = {}
        rerank_event_ids: list[str] = []
        seen: set[str] = set()
        threshold = config.sag2_rerank.rerank_score_threshold
        for result in rerank_results:
            if not isinstance(result, dict):
                continue
            event_id = result.get("id")
            if not event_id or event_id not in event_map or event_id in seen:
                continue
            try:
                score = float(result.get("score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or score < threshold:
                continue
            seen.add(event_id)
            rerank_event_ids.append(event_id)
            rerank_scores[event_id] = score
            if len(rerank_event_ids) >= rerank_top_k:
                break

        rerank_selected_count = len(rerank_event_ids)
        if rerank_selected_count < config.sag2_rerank.max_results:
            similarity_ordered_ids = sorted(
                ordered_event_ids,
                key=lambda event_id: event_scores.get(event_id, 0.0),
                reverse=True,
            )
            for event_id in similarity_ordered_ids:
                if event_id in seen:
                    continue
                if event_scores.get(event_id, 0.0) < config.sag2_rerank.score_threshold:
                    continue
                seen.add(event_id)
                rerank_event_ids.append(event_id)
                if len(rerank_event_ids) >= config.sag2_rerank.max_results:
                    break

        rerank_events = build_ranked_event_results(
            rerank_event_ids, event_map, event_scores, rerank_scores, "rerank_score"
        )
        logger.info(
            "[SAG2] rerank 完成: input=%d, rerank_top_k=%d, rerank_selected=%d, "
            "embedding_filled=%d, output=%d",
            len(documents),
            rerank_top_k,
            rerank_selected_count,
            len(rerank_event_ids) - rerank_selected_count,
            len(rerank_event_ids),
        )
        return rerank_events, rerank_event_ids, rerank_scores, False, event_map


    async def _llm_rank_events(
        self,
        query: str,
        event_ids: list[str],
        event_scores: dict[str, float],
        source_config_ids: list[str],
        config: SAGConfig,
        event_details: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        dict[str, float],
        bool,
        dict[str, dict[str, Any]],
        list[str],
        str | None,
        int,
    ]:
        """LLM rerank using the SAG2-local prompt and schema.

        返回值末尾的 display_ids 是"LLM 真正选中"的子集，不含相似度补齐部分。
        """
        event_ids = merge_ids(event_ids)[: config.sag2_rerank.llm_rank_top_n]
        if not event_ids:
            return [], [], {}, False, {}, [], None, 0

        ordered_event_ids, event_map = await self._load_rank_candidate_events(
            event_ids, source_config_ids, event_details
        )
        if not ordered_event_ids:
            return [], [], {}, False, event_map, [], None, 0

        fallback_ids = ordered_event_ids[: config.sag2_rerank.max_results]

        # 构造候选关系列表（[index] content 截断）
        relation_lines: list[str] = []
        for index, event_id in enumerate(ordered_event_ids):
            event = event_map[event_id]
            if config.sag2_rerank.llm_rank_include_content:
                text = str(event.get("content") or "").strip()
                if text:
                    max_len = config.sag2_rerank.llm_rank_max_content_len
                    text = text[:max_len]
                else:
                    text = str(event.get("title") or event_id).strip()
            else:
                text = str(event.get("title") or event_id).strip()
            relation_lines.append(f"[{index}] {text}")
        relations_str = "\n".join(relation_lines)

        top_k = min(config.sag2_rerank.llm_rank_max_results, len(ordered_event_ids))
        llm_client = None
        try:
            messages = self.runtime.get_prompts(config).get_sag2_rerank_messages(
                question=query,
                relations=relations_str,
                top_k=top_k,
            )
            llm_client = await self.runtime.get_llm_client()
            parsed: SAG2RerankOutput = await llm_client.chat_parsed(messages, SAG2RerankOutput)
            self.timing.record_llm_call_metrics()
        except Exception as exc:
            timing = getattr(llm_client, "_last_call_timing", {}) if llm_client else {}
            retry_count = int(timing.get("retries", 0))
            if timing:
                self.timing.record_llm_call_metrics()
            failure_reason = "chat_parsed_failed" if timing else "llm_rank_setup_failed"
            logger.warning("[SAG2] LLM rank 失败，降级为相似度排序: %s", exc)
            return (
                build_ranked_event_results(
                    fallback_ids, event_map, event_scores, {}, "llm_rank_score"
                ),
                fallback_ids,
                {},
                True,
                event_map,
                [],
                failure_reason,
                retry_count,
            )

        # 解析 useful_relations（形如 "[index]"）为 event_id
        llm_event_ids: list[str] = []
        seen: set[str] = set()
        for line in parsed.useful_relations:
            line = str(line)
            if "[" not in line or "]" not in line:
                continue
            idx_str = line[line.find("[") + 1 : line.find("]")].strip()
            try:
                index = int(idx_str)
            except (ValueError, TypeError):
                continue
            if index < 0 or index >= len(ordered_event_ids):
                continue
            event_id = ordered_event_ids[index]
            if event_id in seen:
                continue
            seen.add(event_id)
            llm_event_ids.append(event_id)
            if len(llm_event_ids) >= config.sag2_rerank.max_results:
                break

        # LLM 真正选中的子集（供前端展示，不含下面的相似度补齐部分）
        llm_selected_ids = list(llm_event_ids)

        # 不足 max_results 时按相似度补齐（过 score_threshold）
        if len(llm_event_ids) < config.sag2_rerank.max_results:
            for event_id in ordered_event_ids:
                if event_id in seen:
                    continue
                if event_scores.get(event_id, 0.0) < config.sag2_rerank.score_threshold:
                    continue
                seen.add(event_id)
                llm_event_ids.append(event_id)
                if len(llm_event_ids) >= config.sag2_rerank.max_results:
                    break

        if not llm_event_ids:
            logger.warning("[SAG2] LLM rank 未解析出有效事件，降级为相似度排序")
            return (
                build_ranked_event_results(
                    fallback_ids, event_map, event_scores, {}, "llm_rank_score"
                ),
                fallback_ids,
                {},
                True,
                event_map,
                [],
                "invalid_llm_selection",
                int(getattr(llm_client, "_last_call_timing", {}).get("retries", 0)),
            )

        llm_events = build_ranked_event_results(
            llm_event_ids, event_map, event_scores, {}, "llm_rank_score"
        )
        logger.info(
            "[SAG2] LLM rank 完成: input=%d, output=%d, llm_selected=%d",
            len(ordered_event_ids),
            len(llm_event_ids),
            len(llm_selected_ids),
        )
        return (
            llm_events,
            llm_event_ids,
            {},
            False,
            event_map,
            llm_selected_ids,
            None,
            int(getattr(llm_client, "_last_call_timing", {}).get("retries", 0)),
        )


    async def _rrf_rank_events(
        self,
        query: str,
        event_ids: list[str],
        event_scores: dict[str, float],
        source_config_ids: list[str],
        config: SAGConfig,
        event_details: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, float], bool, dict[str, dict[str, Any]]]:
        """RRF：query 相似度排名 + ES BM25 排名做名次融合。"""
        event_ids = merge_ids(event_ids)
        if not event_ids:
            return [], [], {}, False, {}

        ordered_event_ids, event_map = await self._load_rank_candidate_events(
            event_ids, source_config_ids, event_details
        )
        if not ordered_event_ids:
            return [], [], {}, False, event_map

        # BM25 remains part of SAG2 RRF when event details come from scope.
        # The scope limits event_ids passed to the selected backend; it must not remove a
        # ranking signal.
        bm25_list = await self.runtime.search_events_by_text(
            query=query,
            event_ids=ordered_event_ids,
            k=len(ordered_event_ids),
            source_config_ids=source_config_ids,
        )
        bm25_scores = {
            r.get("event_id"): float(r.get("_score") or 0.0) for r in bm25_list if r.get("event_id")
        }

        rrf_k = config.sag2_rerank.rrf_k
        sim_ranked = sorted(ordered_event_ids, key=lambda e: event_scores.get(e, 0.0), reverse=True)
        sim_rank = {e: rank for rank, e in enumerate(sim_ranked, start=1)}

        bm25_ranked = sorted(bm25_scores, key=lambda e: bm25_scores[e], reverse=True)
        default_rank = len(ordered_event_ids) + 1
        bm25_rank = {e: rank for rank, e in enumerate(bm25_ranked, start=1)}

        rrf_scores: dict[str, float] = {}
        for e in ordered_event_ids:
            s_rank = sim_rank.get(e, default_rank)
            b_rank = bm25_rank.get(e, default_rank)
            rrf_scores[e] = 1.0 / (rrf_k + s_rank) + 1.0 / (rrf_k + b_rank)

        rrf_sorted = sorted(ordered_event_ids, key=lambda e: rrf_scores[e], reverse=True)[
            : config.sag2_rerank.max_results
        ]
        rrf_events = build_ranked_event_results(
            rrf_sorted, event_map, event_scores, rrf_scores, "rrf_score"
        )
        logger.info(
            "[SAG2] RRF rank 完成: input=%d, bm25_hit=%d, output=%d",
            len(ordered_event_ids),
            len(bm25_scores),
            len(rrf_sorted),
        )
        return rrf_events, rrf_sorted, rrf_scores, False, event_map


__all__ = ["SAG2RerankStage"]
