"""
搜索器 - 统一入口

按 RerankStrategy 路由 VECTOR、ATOMIC、MULTI_ES、SAG 和 BM25。
每种策略读取自己的 strategy_config，并统一返回 sections / clues / stats / query。
"""

import time
from typing import Any

from pipeline.core.prompt.manager import PromptManager
from pipeline.exceptions import SearchError
from pipeline.modules.search.atomic import AtomicSearcher
from pipeline.modules.search.bm25 import BM25ChunkSearcher
from pipeline.modules.search.config import (
    MultiConfig,
    RerankStrategy,
    ReturnType,
    SAGConfig,
    SearchConfig,
)
from pipeline.modules.search.vector import VectorSearcher
from pipeline.utils import get_logger

logger = get_logger("search.searcher")


class SAGSearcher:
    """
    搜索策略统一入口。

    VECTOR、ATOMIC、BM25 使用独立搜索器；MULTI_ES 使用 MultiConfig；
    SAG 使用 SAGConfig，并将 SAG 的领域结果转换成统一 section 结构。

    返回结果格式：
    {
        "sections": List[Dict],        # 段落列表
        "clues": List[Dict],           # 线索列表（支持前端图谱）
        "stats": Dict,                 # 统计信息
        "query": Dict                  # 查询信息
    }
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        model_config: dict | None = None,
    ):
        """
        初始化搜索器

        Args:
            prompt_manager: 提示词管理器
            model_config: LLM配置字典（可选）
        """
        self.prompt_manager = prompt_manager
        self.model_config = model_config
        self.logger = get_logger("search.sag")

        # 轻量搜索器立即初始化，较重的图搜索器按策略延迟初始化。
        self._vector_searcher = VectorSearcher()
        self._atomic_searcher = AtomicSearcher()
        self._bm25_searcher = BM25ChunkSearcher()
        self._multi_es_searcher: Any | None = None
        self._sag_searcher: Any | None = None

        self.logger.info("SAG搜索器初始化完成")

    async def aclose(self) -> None:
        """Close the lazily created SAG searcher, if this instance owns one."""

        sag_searcher = self._sag_searcher
        self._sag_searcher = None
        if sag_searcher is not None:
            await sag_searcher.aclose()

    def _get_multi_es_searcher(self, config: MultiConfig) -> Any:
        if self._multi_es_searcher is None:
            from pipeline.modules.search.multi_vector import MultiSearcherES as ESMultiSearcher

            self._multi_es_searcher = ESMultiSearcher(config=config)
        return self._multi_es_searcher

    def _get_sag_searcher(self) -> Any:
        if self._sag_searcher is None:
            from pipeline.modules.search.sag_graph import SAGGraphSearcher

            self._sag_searcher = SAGGraphSearcher()
        return self._sag_searcher

    def _get_multi_es_config(self, config: SearchConfig) -> MultiConfig:
        if isinstance(config.strategy_config, MultiConfig):
            return config.strategy_config
        if isinstance(config.strategy_config, dict):
            return MultiConfig(**config.strategy_config)
        return MultiConfig()

    def _get_sag_config(self, config: SearchConfig) -> SAGConfig:
        """Extract SAGConfig from strategy_config with backwards-compat for old callers.

        - Already SAGConfig: return as-is.
        - dict: validate via SAGConfig(**dict).
        - MultiConfig (legacy): extract only SAG-relevant fields.
        - None / other: return SAGConfig() defaults.
        """
        import warnings

        sc = config.strategy_config
        if isinstance(sc, SAGConfig):
            return sc
        if isinstance(sc, dict):
            return SAGConfig(**sc)
        if isinstance(sc, MultiConfig):
            warnings.warn(
                "Passing MultiConfig for SAG is deprecated. Use SAGConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return SAGConfig(
                max_sections=sc.max_sections,
                use_mlflow_prompts=sc.use_mlflow_prompts,
                mlflow_prompt_alias=sc.mlflow_prompt_alias,
                mlflow_tracking_uri=sc.mlflow_tracking_uri,
                sag_recall=sc.sag_recall,
                sag_scope=sc.sag_scope,
                sag_expand=sc.sag_expand,
                sag_rerank=sc.sag_rerank,
                sag_rewrite_query_enabled=sc.sag_rewrite_query_enabled,
                sag_enable_entity_extraction=sc.sag_enable_entity_extraction,
                sag_use_fast_mode=sc.sag_use_fast_mode,
            )
        return SAGConfig()

    @staticmethod
    def _get_sag_event_stats(sag_result: dict[str, Any]) -> dict[str, Any]:
        """Read the stable event payload with compatibility for older SAG results."""

        direct = sag_result.get("event_stats")
        if isinstance(direct, dict):
            return direct
        legacy = sag_result.get("stats")
        if not isinstance(legacy, dict):
            return {}
        nested = legacy.get("sag")
        if isinstance(nested, dict):
            return nested
        # Current pre-v2 SAG returned the event counters as a flat ``stats``
        # mapping.  Preserve that payload instead of looking for a nonexistent
        # stats["sag"] child.
        return legacy

    async def search(self, config: SearchConfig) -> dict[str, Any]:
        """
        执行搜索

        Args:
            config: 搜索配置

        Returns:
            {
                "sections": List[Dict],        # 段落列表
                "clues": List[Dict],           # 线索列表
                "stats": Dict,                 # 统计信息
                "query": Dict                  # 查询信息
            }
        """
        try:
            total_start = time.perf_counter()

            # 打印配置参数
            self.logger.info("=" * 100)
            self.logger.info("📋 SAG搜索配置参数详情:")
            self.logger.info("=" * 100)
            self.logger.info("🔹 基础参数:")
            self.logger.info(f"  query: '{config.query}'")
            self.logger.info(f"  strategy: {config.rerank.strategy}")
            self.logger.info(
                f"  source_config_ids: {config.source_config_ids[:5] if config.source_config_ids else []}"
            )
            # return_type 仅 VECTOR 策略生效（且实际生效值在 strategy_config.return_type）；
            # ATOMIC/MULTI_ES 固定返回段落，忽略该字段，故不打印以免误导
            if config.rerank.strategy == RerankStrategy.VECTOR:
                effective_return_type = getattr(
                    config.strategy_config, "return_type", config.return_type
                )
                self.logger.info(f"  return_type: {effective_return_type} (VECTOR 生效)")
            self.logger.info("=" * 100)

            self.logger.info(
                f"🔍 开始搜索：query='{config.query}', strategy={config.rerank.strategy}"
            )

            # 根据策略选择搜索方式
            strategy = config.rerank.strategy

            # VECTOR 模式：纯向量搜索
            if strategy == RerankStrategy.VECTOR:
                if config.return_type != ReturnType.PARAGRAPH:
                    raise SearchError(
                        f"VECTOR 策略仅支持 PARAGRAPH 模式，当前为 {config.return_type}"
                    )

                self.logger.info("=" * 60)
                self.logger.info("【VECTOR 模式】跳过 Recall/Expand，直接向量搜索")
                self.logger.info("=" * 60)

                vector_start = time.perf_counter()
                rerank_result = await self._vector_searcher.search_chunks_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                vector_time = time.perf_counter() - vector_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "vector": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "vector": vector_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ VECTOR 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # BM25 模式：纯关键词召回 chunk
            elif strategy == RerankStrategy.BM25:
                self.logger.info("=" * 60)
                self.logger.info("【BM25 模式】跳过 Recall/Expand，纯关键词召回 chunk")
                self.logger.info("=" * 60)

                bm25_start = time.perf_counter()
                rerank_result = await self._bm25_searcher.search_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                bm25_time = time.perf_counter() - bm25_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "bm25": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "bm25": bm25_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ BM25 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # ATOMIC 模式：原子事项检索
            elif strategy == RerankStrategy.ATOMIC:
                self.logger.info("=" * 60)
                self.logger.info("【ATOMIC 模式】原子事项检索")
                self.logger.info("=" * 60)

                atomic_start = time.perf_counter()
                rerank_result = await self._atomic_searcher.search_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                atomic_time = time.perf_counter() - atomic_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "atomic": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "atomic": atomic_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ ATOMIC 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # MULTI_ES 模式：ES-first 多元事项检索
            elif strategy == RerankStrategy.MULTI_ES:
                multi_config = self._get_multi_es_config(config)

                self.logger.info("=" * 60)
                self.logger.info(f"【MULTI_ES 模式】ES 多元事项检索，mode={multi_config.mode}")
                self.logger.info("=" * 60)

                multi_start = time.perf_counter()
                multi_es_searcher = self._get_multi_es_searcher(multi_config)
                await multi_es_searcher.warmup(multi_config)
                rerank_result = await multi_es_searcher.search_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=multi_config,
                )
                multi_time = time.perf_counter() - multi_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "multi_es": {
                            "sections_count": len(rerank_result.get("sections", [])),
                            "mode": multi_config.mode,
                        },
                        "timing": {
                            "multi_es": multi_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ MULTI_ES 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # SAG 模式：独立 SAG 搜索器（并行两路召回 + 相似度过滤 + 多跳扩展 + route 追链）
            elif strategy == RerankStrategy.SAG:
                sag_config = self._get_sag_config(config)
                target_k = sag_config.max_sections
                self.logger.info("=" * 60)
                self.logger.info("【SAG 模式】并行两路召回 + 相似度过滤 + 多跳扩展 + route 追链")
                self.logger.info(
                    "召回配置: max_entities=%d, query_recall_event_max=%d, max_events_per_key=%d, score_threshold=%.2f",
                    sag_config.sag_recall.max_entities,
                    sag_config.sag_recall.query_recall_event_max,
                    sag_config.sag_recall.max_events_per_key,
                    sag_config.sag_recall.score_threshold,
                )
                self.logger.info(
                    "扩展配置: enabled=%s, max_hops=%d, entities_per_hop=%d, max_events_per_hop=%d, seed_event_limit=%d",
                    sag_config.sag_expand.enabled,
                    sag_config.sag_expand.max_hops,
                    sag_config.sag_expand.entities_per_hop,
                    sag_config.sag_expand.max_events_per_hop,
                    sag_config.sag_expand.seed_event_limit,
                )
                self.logger.info(
                    "排序配置: strategy=%s, max_results=%d",
                    sag_config.sag_rerank.strategy,
                    sag_config.sag_rerank.max_results,
                )
                self.logger.info("=" * 60)

                sag_start = time.perf_counter()
                sag_searcher = self._get_sag_searcher()
                sag_result = await sag_searcher.search(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=sag_config,
                    gold_evidences=config.gold_evidences,
                )
                sag_time = time.perf_counter() - sag_start
                event_stats = self._get_sag_event_stats(sag_result)

                # 将 SAG 领域结果中的 chunk 转换为统一 section 结构。
                items = sag_result.get("items", [])
                sag_sections = []
                seen_chunk_ids: set[str] = set()
                for item in items:
                    chunk = item.get("chunk")
                    if not chunk:
                        continue
                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)
                    sag_sections.append(
                        {
                            "chunk_id": chunk_id,
                            "source_id": chunk.get("source_id", ""),
                            "source_config_id": chunk.get("source_config_id", ""),
                            "heading": chunk.get("heading", ""),
                            "content": chunk.get("content", ""),
                            "score": item.get("score", 0.0),
                            "weight": item.get("score", 0.0),
                        }
                    )

                total_time = time.perf_counter() - total_start
                response = {
                    "sections": sag_sections[:target_k],
                    "clues": sag_result.get("clues", []),
                    "nodes": sag_result.get("nodes", {}),
                    "all_clues": sag_result.get("all_clues", []),
                    "all_nodes": sag_result.get("all_nodes", {}),
                    "route_index": sag_result.get("route_index", {}),
                    "evidence_coverage": sag_result.get("evidence_coverage", {}),
                    "rank_event_ids": sag_result.get("rank_event_ids", []),
                    "display_event_ids": sag_result.get("display_event_ids", []),
                    "rank_scores": sag_result.get("rank_scores", {}),
                    "score_top_event_ids": sag_result.get("score_top_event_ids", []),
                    "expand_hops": sag_result.get("expand_hops", []),
                    "recall_clue_stats": sag_result.get("recall_clue_stats", {}),
                    "chunk_headings": sag_result.get("chunk_headings", {}),
                    "stats": {
                        "event_stats": dict(event_stats),
                        # Backward-compatible alias for callers that still
                        # inspect response["stats"]["sag"].
                        "sag": dict(event_stats),
                        "route_stats": sag_result.get("route_stats", {}),
                        "timing": {
                            "sag": sag_time,
                            "total": total_time,
                        },
                        "timing_steps": sag_result.get("_timings", {}),
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    "✅ SAG 搜索完成：返回 %d 个段落，%d 条 clues，总耗时=%.3fs",
                    len(response["sections"]),
                    len(response.get("clues", [])),
                    total_time,
                )
                return response

            else:
                raise SearchError(f"不支持的搜索策略: {strategy}")

        except Exception as e:
            self.logger.error(f"❌ 搜索失败: {e}", exc_info=True)
            raise SearchError(f"搜索失败: {e}") from e


# 向后兼容别名
EventSearcher = SAGSearcher

__all__ = [
    "SAGSearcher",
    "EventSearcher",
]
