"""SAG2 独立搜索器。

召回阶段（两路并行）：
1. 路 A：query -> embedding -> event，直接召回事项候选，返回 event_id / entity_ids / score，
   并产出全量相似度缓存 event_id -> score。
2. 路 B：query -> BM25 -> entities -> event_ids（不打分），与路 A 在 asyncio.gather 中并行；
   随后做相似度过滤，复用路 A 的分数缓存（关系映射不丢），仅对缓存外的 event 发起打分。

SAG2Runtime 负责客户端、存储端口、提示词和 chunk hydration；SAG2Searcher
负责召回、扩展、排序、证据追踪与结果组装。可选 candidate scope 只限定
请求内候选域，不改变其余 SAG2 执行链路。
"""

import time
import uuid
from typing import Any

from pipeline.modules.search.config import SAGConfig
from pipeline.utils import get_logger

from . import utils as sag2_utils
from .contracts import (
    SAG2_TIMING_STAGE_ORDER,
    SAG2Request,
    SAG2SearchState,
)
from .evidence import EvidenceTracker
from .expand import SAG2ExpandStage
from .recall import SAG2RecallStage
from .rerank import SAG2RerankStage
from .routes import SAG2RouteTracker
from .runtime import SAG2Runtime
from .timing import SAG2TimingService

logger = get_logger("search.sag2")


class SAG2Searcher:
    """SAG2 search entrypoint with infrastructure provided by SAG2Runtime."""

    def __init__(
        self,
        runtime: SAG2Runtime | None = None,
        *,
        rerank_client=None,
        settings=None,
    ):
        self._runtime = runtime or SAG2Runtime()
        self._rerank_client = rerank_client
        self._settings = settings
        self._timing_service = SAG2TimingService(
            self._runtime,
            ctx_var_name=f"sag2_step7_llm_timing_{id(self)}",
        )
        self._recall_stage: SAG2RecallStage = SAG2RecallStage(
            runtime=self._runtime,
            timing=self._timing_service,
        )
        self._expand_stage: SAG2ExpandStage = SAG2ExpandStage(
            runtime=self._runtime,
            timing=self._timing_service,
        )
        self._rerank_stage: SAG2RerankStage = SAG2RerankStage(
            runtime=self._runtime,
            timing=self._timing_service,
            rerank_client=self._rerank_client,
            settings=self._settings,
        )

    async def aclose(self) -> None:
        """Release resources owned by this searcher's SAG2 runtime."""

        await self._runtime.aclose()

    # ------------------------------------------------------------------
    # SAG2-owned dependency accessors
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 主搜索方法
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        source_config_ids: list[str],
        config: SAGConfig | None = None,
        *,
        gold_evidences: list[str] | None = None,
    ) -> dict[str, Any]:
        """执行 SAG2 搜索。

        Args:
            query: 查询文本。
            source_config_ids: 数据源 ID 列表。
            config: SAGConfig 配置（读取 sag2_recall/sag2_expand/sag2_rerank）。
            gold_evidences: 可选，该问题的 gold evidence ID 列表，传入后在每个
                           召回/过滤事件阶段打印证据命中数/总数。

        Returns:
            {
                "items": [{"event_id","title","summary","content","score","chunk"}, ...],
                "_timings": {..., "total": float},
                # + SAG2 诊断字段（route_stats/event_scores/rank_method/...）
                # + evidence_coverage（gold_evidences 传入时）
            }
        """
        config = config or SAGConfig()
        if not source_config_ids:
            raise ValueError("SAG2 search 必须传入 source_config_ids")

        total_start = time.perf_counter()
        self._runtime.get_prompts(config)
        # 按 config 开关一次性加载提示词（委托 SAG2Runtime 的实例级缓存）

        tracker = EvidenceTracker(set(gold_evidences or []))

        state = SAG2SearchState()
        timings: dict[str, float] = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
        retry_wasted_by_stage: dict[str, float] = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
        # ContextVar makes the request accumulator task-local even though
        # SAGSearcher intentionally caches and concurrently reuses this
        # SAG2Searcher instance.
        timing_context_token = self._timing_service.start_request()

        try:
            logger.info(
                "[SAG2] 搜索开始: query=%r, rerank_strategy=%s, rewrite=%s, ner=%s, fast=%s",
                query,
                config.sag2_rerank.strategy,
                config.sag2_rewrite_query_enabled,
                config.sag2_enable_entity_extraction,
                config.sag2_use_fast_mode,
            )

            route_index = SAG2RouteTracker.new_route_index(query)

            recall_result = await self._recall_stage.run(
                SAG2Request(
                    query=query,
                    source_config_ids=source_config_ids,
                    config=config,
                    gold_evidences=list(gold_evidences or []),
                ),
                state=state,
                route_index=route_index,
                evidence=tracker,
                timings=timings,
                retry_wasted_by_stage=retry_wasted_by_stage,
            )
            rewritten_query = recall_result.rewritten_query
            rewritten_entities = recall_result.rewritten_entities
            scope = recall_result.scope
            query_events = recall_result.query_events
            entity_events = recall_result.entity_events
            query_event_ids = recall_result.query_event_ids
            entity_event_ids = recall_result.entity_event_ids
            query_entity_ids = recall_result.query_entity_ids
            event_ids = recall_result.event_ids
            event_scores = recall_result.event_scores
            route_stats = recall_result.initial_route_stats

            # ---- expand（多跳）----
            expand_result = await self._expand_stage.run(
                request=SAG2Request(
                    query=query,
                    source_config_ids=source_config_ids,
                    config=config,
                    gold_evidences=list(gold_evidences or []),
                ),
                recall=recall_result,
                state=state,
                route_index=route_index,
                evidence=tracker,
                timings=timings,
                retry_wasted_by_stage=retry_wasted_by_stage,
            )
            event_ids = expand_result.event_ids
            event_scores = expand_result.event_scores
            expand_event_ids = expand_result.expand_event_ids
            new_entity_ids = expand_result.new_entity_ids
            expand_new_entity_ids = expand_result.expand_new_entity_ids
            expand_hops = expand_result.expand_hops
            route_stats.update(expand_result.route_stats)

            # ---- score top-100 ----
            t0 = time.perf_counter()
            score_top_event_ids = sag2_utils.top_event_ids_by_score(event_scores, limit=100)
            self._timing_service.record_timed_stage(
                timings, retry_wasted_by_stage, "score_sort", t0
            )
            tracker.check("5_score_top100", score_top_event_ids)

            # ---- 排序（llm_rank 默认 / rerank / rrf）----
            rerank_result = await self._rerank_stage.run(
                request=SAG2Request(
                    query=query,
                    source_config_ids=source_config_ids,
                    config=config,
                    gold_evidences=list(gold_evidences or []),
                ),
                event_ids=score_top_event_ids,
                event_scores=event_scores,
                event_details=scope.event_details if scope is not None else None,
                evidence=tracker,
                timings=timings,
                retry_wasted_by_stage=retry_wasted_by_stage,
            )
            rank_events = rerank_result.rank_events
            rank_event_ids = rerank_result.rank_event_ids
            rank_scores = rerank_result.rank_scores
            rank_failed = rerank_result.rank_failed
            rank_method = rerank_result.rank_method
            rank_event_map = rerank_result.rank_event_map
            display_event_ids = rerank_result.display_event_ids
            rank_failure_reason = rerank_result.failure_reason
            rank_retry_count = rerank_result.retry_count

            # ---- route 反向追链 ----
            t0 = time.perf_counter()
            answer_clue_stats = await self._build_answer_clues_from_routes(
                state=state,
                route_index=route_index,
                answer_event_ids=rank_event_ids,
                event_map=rank_event_map,
                entity_map=(
                    {
                        entity_id: {"entity_id": entity_id, "name": entity_id}
                        for entity_id in scope.entity_ids
                    }
                    if scope is not None
                    else None
                ),
            )
            self._timing_service.record_timed_stage(
                timings, retry_wasted_by_stage, "answer_graph", t0
            )
            route_stats["total_edges"] = route_index.get("edge_count", 0)
            route_stats["node_count"] = len(route_index.get("node_hops", {}))

            # ---- chunk 补齐 ----
            t0 = time.perf_counter()
            # 1) 最终 rank events 的 chunk（组装 items 用）
            rank_event_id_list = [i["event_id"] for i in rank_events]
            fetch_ids = rank_event_id_list
            chunk_map = await self._runtime.fetch_event_chunks(fetch_ids)
            # 2) EvidenceTracker 各阶段全量 event_id 的 chunk（覆盖率独立计算用）
            # 仅当有 gold 需要计算覆盖率时才补查 ES（避免无 gold 时白查）
            if tracker.total_evidence > 0:
                all_tracker_ids = tracker.all_event_ids()
                tracker_chunk_map = (
                    await self._runtime.fetch_event_chunks(all_tracker_ids)
                    if all_tracker_ids
                    else {}
                )
                full_chunk_map = {**chunk_map, **tracker_chunk_map}
                self._timing_service.record_timed_stage(
                    timings, retry_wasted_by_stage, "step8_chunks", t0
                )

                # ---- 用 chunk 文本匹配 gold paragraph，更新覆盖率 ----
                tracker.resolve(full_chunk_map)
            else:
                self._timing_service.record_timed_stage(
                    timings, retry_wasted_by_stage, "step8_chunks", t0
                )

            # ---- 组装 items（补 title/summary/content/chunk）----
            finalize_t0 = time.perf_counter()
            items = self._build_items(rank_events, rank_event_map, chunk_map)

            # event_id → chunk.heading 映射（追链展示用，复用已有 full_chunk_map，不额外查库）
            _chunk_src = full_chunk_map if tracker.total_evidence > 0 else chunk_map
            chunk_headings = {eid: c.get("heading", "") for eid, c in _chunk_src.items()}
            cov_summary = tracker.summary()
            event_stats = self._timing_service.build_event_stats(
                query_events=query_events,
                entity_events=entity_events,
                expand_event_ids=expand_event_ids,
                event_ids=event_ids,
                score_top_event_ids=score_top_event_ids,
                rank_event_ids=rank_event_ids,
                seen_event_ids=state.seen_event_ids,
                query_entity_ids=query_entity_ids,
                new_entity_ids=new_entity_ids,
                scope=scope,
            )
            self._timing_service.record_timed_stage(
                timings, retry_wasted_by_stage, "finalize", finalize_t0
            )

            wall_total_observed = time.perf_counter() - total_start
            explicit_total = sum(
                timings[stage] for stage in SAG2_TIMING_STAGE_ORDER if stage != "other_overhead"
            )
            timings["other_overhead"] = max(0.0, wall_total_observed - explicit_total)
            step7_t = self._timing_service.context_var.get() or {}
            timing_payload = self._timing_service.build_timing_payload(
                timings,
                retry_wasted_by_stage,
                wall_total_observed,
                step7_t,
            )
            logger.info(
                "[SAG2] 搜索完成 total=%.2fs, items=%d, rank_method=%s",
                timing_payload["total_with_retry"],
                len(items),
                rank_method,
            )
            if cov_summary["total_evidence"] > 0:
                logger.info(
                    "[evidence] 各阶段覆盖率: %s",
                    " -> ".join(
                        f"{s['stage']}={s['hit']}/{s['total_evidence']}"
                        for s in cov_summary["stages"]
                    ),
                )
            return {
                "items": items,
                "_timings": timing_payload,
                # ── SAG2 诊断字段 ──
                "rewritten_query": rewritten_query,
                "rewritten_entities": [e.model_dump() for e in rewritten_entities],
                "event_ids": event_ids,
                "event_scores": event_scores,
                "score_top_event_ids": score_top_event_ids,
                "rank_method": rank_method,
                "rank_event_ids": rank_event_ids,
                "display_event_ids": display_event_ids,
                "rank_scores": rank_scores,
                "rank_failed": rank_failed,
                "rank_failure_reason": rank_failure_reason,
                "rank_retry_count": rank_retry_count,
                "route_stats": route_stats,
                "route_index": route_index,
                "clues": state.all_clues,
                "nodes": state.all_nodes,
                "chunk_headings": chunk_headings,
                "recall_clue_stats": answer_clue_stats,
                "query_event_ids": query_event_ids,
                "entity_event_ids": entity_event_ids,
                "query_entity_ids": query_entity_ids,
                "new_entity_ids": new_entity_ids,
                "expand_event_ids": expand_event_ids,
                "expand_new_entity_ids": expand_new_entity_ids,
                "expand_hops": expand_hops,
                "event_stats": dict(event_stats),
                "stats": dict(event_stats),
                "candidate_scope_enabled": scope is not None,
                "candidate_scope_event_count": len(scope.event_scores) if scope is not None else 0,
                "candidate_scope_entity_count": len(scope.entity_ids) if scope is not None else 0,
                "candidate_scope_edge_count": (
                    sum(len(values) for values in scope.event_to_entities.values())
                    if scope is not None
                    else 0
                ),
                "evidence_coverage": cov_summary,
            }
        except Exception as exc:
            logger.error("[SAG2] 搜索失败: %s", exc, exc_info=True)
            raise
        finally:
            self._timing_service.context_var.reset(timing_context_token)

    # ------------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------------

    @staticmethod
    def _build_items(
        rank_events: list[dict[str, Any]],
        event_map: dict[str, dict[str, Any]],
        chunk_map: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        """把排序结果组装为 benchmark 兼容的 items（补 title/summary/content/chunk）。"""
        items: list[dict[str, Any]] = []
        for ev in rank_events:
            event_id = ev["event_id"]
            detail = event_map.get(event_id, {})
            items.append(
                {
                    "event_id": event_id,
                    "title": detail.get("title", "") or "",
                    "summary": detail.get("summary", "") or "",
                    "content": detail.get("content", "") or "",
                    "score": ev.get("score", 0.0),
                    "chunk": chunk_map.get(event_id),
                }
            )
        return items

    # ------------------------------------------------------------------
    # 召回路 A：query -> event
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 召回路 B：query -> entities -> event candidates
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # expand 阶段
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 排序阶段
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # query 重写 + 实体提取
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # route 反向追链系统（自带轻量实现，无 Tracker 依赖）
    # ------------------------------------------------------------------

    async def _build_answer_clues_from_routes(
        self,
        state: SAG2SearchState,
        route_index: dict[str, Any],
        answer_event_ids: list[str],
        event_map: dict[str, dict[str, Any]],
        entity_map: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """从答案事项反向追链，补齐实体信息后写入 state.all_clues / all_nodes。"""
        before_clue_count = len(state.all_clues)
        before_node_count = len(state.all_nodes)

        selected_edges = self._collect_route_edges_for_events(route_index, answer_event_ids)
        needed_entity_ids = sag2_utils.entity_ids_from_route_edges(selected_edges)
        needed_event_ids = sag2_utils.event_ids_from_route_edges(selected_edges)

        # 先批量加载路径实体，clue 节点即可携带完整实体信息（name/description）。
        if entity_map is None:
            entity_map = await self._load_route_entities(needed_entity_ids)
        else:
            entity_map = {
                entity_id: entity_map.get(entity_id, {}) for entity_id in needed_entity_ids
            }

        for edge in selected_edges:
            self._add_clue(
                state=state,
                stage=edge["stage"],
                from_node=self._route_node_to_full_node(
                    route_index, edge["from"], event_map, entity_map
                ),
                to_node=self._route_node_to_full_node(
                    route_index, edge["to"], event_map, entity_map
                ),
                confidence=edge["score"],
                relation=edge["relation"],
                display_level="final",
                metadata={
                    "algorithm": "sag2",
                    "method": edge["method"],
                    "hop": edge["hop"],
                    "from_hop": edge["from_hop"],
                    "to_hop": edge["to_hop"],
                    "from_type": edge["from_type"],
                    "from_id": edge["from_id"],
                    "to_type": edge["to_type"],
                    "to_id": edge["to_id"],
                    "score": edge["score"],
                },
            )

        stats = {
            "answer_events": len(sag2_utils.merge_ids(answer_event_ids)),
            "route_edges": len(selected_edges),
            "needed_entity_ids": needed_entity_ids,
            "needed_entity_count": len(needed_entity_ids),
            "entities_loaded": len(entity_map),
            "needed_event_ids": needed_event_ids,
            "needed_event_count": len(needed_event_ids),
            "total_clues": len(state.all_clues) - before_clue_count,
            "new_nodes": len(state.all_nodes) - before_node_count,
        }
        logger.info(
            "[SAG2] 答案图谱构建完成: answer_events=%d, route_edges=%d, new_clues=%d",
            stats["answer_events"],
            stats["route_edges"],
            stats["total_clues"],
        )
        return stats

    def _add_clue(
        self,
        state: SAG2SearchState,
        stage: str,
        from_node: dict[str, Any],
        to_node: dict[str, Any],
        confidence: float,
        relation: str,
        display_level: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """轻量 clue 写入（复刻 Tracker.add_clue 的去重 + 节点注册）。"""
        from_id = from_node["id"]
        to_id = to_node["id"]
        for clue in state.all_clues:
            if (
                clue["from"] == from_id
                and clue["to"] == to_id
                and clue.get("stage") == stage
                and clue.get("display_level") == display_level
            ):
                return clue
        # 注册节点（去重）
        for node in (from_node, to_node):
            node_id = node.get("id")
            if node_id and node_id not in state.all_nodes:
                state.all_nodes[node_id] = node
        clue = {
            "id": str(uuid.uuid4()),
            "stage": stage,
            "from": from_id,
            "to": to_id,
            "confidence": float(confidence or 0.0),
            "description": "",
            "relation": relation,
            "metadata": metadata or {},
            "display_level": display_level,
        }
        state.all_clues.append(clue)
        return clue

    def _collect_route_edges_for_events(
        self,
        route_index: dict[str, Any],
        event_ids: list[str],
    ) -> list[dict[str, Any]]:
        """从答案 event 反向收集链路边。"""
        incoming = route_index.get("incoming", {})
        query_key = route_index.get("original_query_key") or route_index.get("query_key")
        stack = [
            sag2_utils.route_node_key("event", event_id)
            for event_id in sag2_utils.merge_ids(event_ids)
            if event_id
        ]
        visited_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str, int]] = set()
        selected_edges: list[dict[str, Any]] = []

        while stack:
            node_key = stack.pop()
            if node_key in visited_nodes:
                continue
            visited_nodes.add(node_key)
            if node_key == query_key:
                continue
            for edge in incoming.get(node_key, []):
                edge_identity = (edge["from"], edge["to"], edge["method"], edge["hop"])
                if edge_identity in seen_edges:
                    continue
                seen_edges.add(edge_identity)
                selected_edges.append(edge)
                if edge["from"] != query_key:
                    stack.append(edge["from"])

        selected_edges.sort(
            key=lambda edge: (
                edge.get("hop", 0),
                edge.get("from_hop", 0),
                edge.get("to_hop", 0),
                edge.get("from", ""),
                edge.get("to", ""),
            )
        )
        return selected_edges

    async def _load_route_entities(
        self,
        entity_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """批量加载答案路径中需要的实体信息。"""
        entity_ids = sag2_utils.merge_ids(entity_ids)
        if not entity_ids:
            return {}
        entities = await self._runtime.get_entities_by_ids(entity_ids)
        return {
            (e.get("entity_id") or e.get("id")): e
            for e in entities
            if e.get("entity_id") or e.get("id")
        }

    def _route_node_to_full_node(
        self,
        route_index: dict[str, Any],
        node_key: str,
        event_map: dict[str, dict[str, Any]],
        entity_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """把 route node key 转成完整节点。"""
        node_type, node_id = node_key.split(":", 1)
        hop = route_index.get("node_hops", {}).get(node_key, 0)
        if node_type == "query":
            rewritten_id = route_index.get("rewritten_query_node_id")
            if rewritten_id and node_id == rewritten_id:
                return {
                    "id": node_id,
                    "type": "query",
                    "category": "rewrite",
                    "content": route_index.get("rewritten_query_text", node_id),
                }
            return {
                "id": node_id,
                "type": "query",
                "category": "origin",
                "content": route_index.get("query_text", node_id),
            }
        if node_type == "event":
            node = sag2_utils.full_event_node(node_id, event_map.get(node_id, {}))
            node["hop"] = hop
            node["stage"] = "recall" if hop == 0 else "expand"
            return node
        node = sag2_utils.full_entity_node(node_id, entity_map.get(node_id, {}))
        node["hop"] = hop
        return node


__all__ = ["SAG2Searcher"]
