"""
BM25 关键词检索器

独立于三阶段向量检索，直接对 SourceChunk 做 ES 全文检索（multi_match BM25），
跳过 Recall/Expand/LLM 精排，纯关键词召回 chunk，便于与向量/混合召回横向对比。

使用示例：
    from pipeline.modules.search import BM25ChunkSearcher, BM25Config

    config = BM25Config(
        top_k=20,
        similarity_threshold=0.0,
    )

    searcher = BM25ChunkSearcher()
    result = await searcher.search_for_rerank(
        query="人工智能技术发展",
        source_config_ids=["source_1", "source_2"],
        config=config,
    )
    sections = result["sections"]
"""

import time
from typing import Any

from pipeline.storage.facade import get_storage_facade
from pipeline.storage.interfaces import ChunkTextSearchStore
from pipeline.utils import get_logger

logger = get_logger("search.bm25")


class BM25ChunkSearcher:
    """
    BM25 关键词检索器

    对 source_chunks 索引执行 ES multi_match 全文检索（heading^2 + content），
    返回段落格式（与 VectorSearcher 的 sections 口径一致）。
    """

    def __init__(self, chunk_text_store: ChunkTextSearchStore | None = None):
        """Initialize with an optional chunk keyword-search dependency."""
        self._chunk_text_store = chunk_text_store

    def _get_chunk_text_store(self) -> ChunkTextSearchStore:
        """Resolve the canonical storage port only when BM25 is used."""
        if self._chunk_text_store is None:
            self._chunk_text_store = get_storage_facade().chunk_text
        if self._chunk_text_store is None:
            raise RuntimeError("Chunk text search provider is not configured")
        return self._chunk_text_store

    async def search_for_rerank(
        self,
        query: str,
        source_config_ids: list[str],
        config: Any | None = None,
    ) -> dict[str, Any]:
        """
        执行 BM25 关键词检索，返回段落格式

        保留与 VectorSearcher.search_chunks_for_rerank 对齐的接口名；
        内部仅做 ES 全文 BM25 召回，不生成向量、不做多跳、不做 LLM 精排。

        Args:
            query: 查询文本
            source_config_ids: 信息源ID列表
            config: BM25Config / SearchConfig 对象（用 getattr 容错取参）

        Returns:
            {
                "sections": [...],  # 段落列表，按 BM25 分降序
                "_timings": {...}   # 耗时统计
            }
        """
        start_time = time.perf_counter()

        # 从 BM25Config / SearchConfig 获取参数
        top_k = 20
        min_score = 0.0

        if config:
            top_k = getattr(config, "top_k", top_k)
            min_score = getattr(config, "similarity_threshold", min_score)

        # 空 query 防护：空串/纯标点 query 传给 ES multi_match 会触发 parse 异常，
        # 这里短路返回空结果，避免单条查询抛错拖崩整批 benchmark。
        if not query or not query.strip():
            logger.info(f"【BM25 检索】空 query，直接返回空结果: {query!r}")
            total_time = time.perf_counter() - start_time
            return {
                "sections": [],
                "_timings": {
                    "es_search": 0.0,
                    "total": total_time,
                },
            }

        logger.info("=" * 60)
        logger.info(f"【BM25 检索】Query: '{query}'")
        logger.info(f"  top_k={top_k}, min_score={min_score}")
        logger.info("=" * 60)

        # Step 1: ES 全文 BM25 检索（heading^2 + content）
        es_start = time.perf_counter()
        es_results = await self._get_chunk_text_store().search_chunks_by_text(
            query=query,
            source_config_ids=source_config_ids,
            size=top_k,
        )
        es_time = time.perf_counter() - es_start
        logger.info(f"✓ ES BM25 搜索完成，命中 {len(es_results)} 个段落，耗时={es_time:.3f}s")

        if not es_results:
            logger.info("【BM25 检索】未找到匹配段落")
            total_time = time.perf_counter() - start_time
            return {
                "sections": [],
                "_timings": {
                    "es_search": es_time,
                    "total": total_time,
                },
            }

        # Step 2: 格式化结果（字段与 VectorSearcher 完全一致）
        sections = []
        for result in es_results:
            score = float(result.get("_score", 0.0) or 0.0)
            if score < min_score:
                continue

            sections.append(
                {
                    "chunk_id": result.get("chunk_id"),
                    "source_id": result.get("source_id"),
                    "source_config_id": result.get("source_config_id"),
                    "heading": result.get("heading"),
                    "content": result.get("content"),
                    "rank": result.get("rank"),
                    "score": score,
                    "weight": score,
                }
            )

        # ES 返回本已按 _score 降序，这里显式再排一次保证稳定性
        sections = sorted(sections, key=lambda x: x["score"], reverse=True)
        sections = sections[:top_k]

        total_time = time.perf_counter() - start_time

        logger.info("=" * 60)
        logger.info(f"【BM25 检索】完成，返回 {len(sections)} 个段落，总耗时={total_time:.3f}s")
        logger.info("=" * 60)

        # Top-5 日志
        for i, sec in enumerate(sections[:5]):
            heading = sec.get("heading", "")[:40] if sec.get("heading") else "无标题"
            logger.info(f"  Top-{i + 1}: score={sec['score']:.4f} | {heading}...")

        return {
            "sections": sections,
            "_timings": {
                "es_search": es_time,
                "total": total_time,
            },
        }


__all__ = ["BM25ChunkSearcher"]
