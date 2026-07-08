"""
向量检索器

独立于三阶段的向量检索器，直接使用 Query 向量检索 Event/Chunk，
支持当前配置后端的 content 向量搜索。

使用示例：
    from pipeline.modules.search import VectorSearcher, VectorConfig

    config = VectorConfig(
        return_type="event",
        top_k=20,
        title_weight=0.3,
        content_weight=0.7,
        similarity_threshold=0.4
    )

    searcher = VectorSearcher()
    events = await searcher.search(
        query="人工智能技术发展",
        source_config_ids=["source_1", "source_2"],
        config=config
    )
"""

import time
from typing import Any, Dict, List, Optional, Union

from pipeline.db import get_session_factory
from pipeline.modules.load.processor import DocumentProcessor
from pipeline.modules.search.config import VectorConfig
from pipeline.storage import get_storage_facade
from pipeline.utils import get_logger

logger = get_logger("search.vector")


class VectorSearcher:
    """
    向量检索器

    支持段落(SourceChunk)向量搜索。
    通过 StorageFacade 隐藏 Elasticsearch / OceanBase 查询差异。
    """

    INDEX_EVENTS = "event_vectors"
    INDEX_CHUNKS = "source_chunks"

    def __init__(self):
        """初始化向量检索器"""
        self.vector_store = get_storage_facade().vector
        self.session_factory = get_session_factory()
        self.processor = DocumentProcessor()

    async def search_chunks_for_rerank(
        self,
        query: str,
        source_config_ids: List[str],
        query_vector: Optional[List[float]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        执行向量搜索，返回段落格式

        保留旧 SAGSearcher 需要的接口名；内部只做向量召回。
        使用当前配置的向量后端检索。

        Args:
            query: 查询文本
            source_config_ids: 信息源ID列表
            query_vector: 可选的预计算向量（避免重复计算）
            config: SearchConfig 对象

        Returns:
            {
                "sections": [...],  # 段落列表，按相似度降序
                "_timings": {...}   # 耗时统计
            }
        """
        start_time = time.perf_counter()

        # 从 VectorConfig / SearchConfig 获取参数
        top_k = 20
        min_score = 0.0

        if config:
            top_k = getattr(config, "top_k", top_k)
            min_score = getattr(config, "similarity_threshold", min_score)

        logger.info("=" * 60)
        logger.info(f"【向量检索】Query: '{query}'")
        logger.info(
            f"  top_k={top_k}, min_score={min_score}"
        )
        logger.info("=" * 60)

        # Step 1: 生成查询向量
        vector_time = 0.0
        if query_vector is None:
            vector_start = time.perf_counter()
            query_vector = await self.processor.generate_embedding(query)
            vector_time = time.perf_counter() - vector_start
            logger.info(f"✓ 向量生成完成，维度={len(query_vector)}，耗时={vector_time:.3f}s")
        else:
            logger.info(f"✓ 使用预计算向量，维度={len(query_vector)}")

        # Step 2: 使用当前配置的向量/检索后端
        vector_start = time.perf_counter()
        vector_results = await self.vector_store.search_chunks_by_vector(
            query_vector=query_vector,
            k=top_k,
            source_config_ids=source_config_ids,
        )
        vector_search_time = time.perf_counter() - vector_start
        logger.info(
            "✓ 向量搜索完成，backend=%s，命中 %s 个段落，耗时=%.3fs",
            self.vector_store.backend_name,
            len(vector_results),
            vector_search_time,
        )

        if not vector_results:
            logger.info("【向量检索】未找到匹配段落")
            total_time = time.perf_counter() - start_time
            return {
                "sections": [],
                "_timings": {
                    "vector_gen": vector_time,
                    "vector_search": vector_search_time,
                    "total": total_time,
                }
            }

        # Step 3: 格式化结果（后端返回统一字段）
        sections = []
        for result in vector_results:
            score = result.get("_score", 0.0)
            if score < min_score:
                continue

            sections.append({
                "chunk_id": result.get("chunk_id"),
                "source_id": result.get("source_id"),
                "source_config_id": result.get("source_config_id"),
                "heading": result.get("heading"),
                "content": result.get("content"),
                "rank": result.get("rank"),
                "score": score,
                "weight": score,
            })

        sections = sorted(sections, key=lambda x: x["score"], reverse=True)

        sections = sections[:top_k]

        total_time = time.perf_counter() - start_time

        logger.info("=" * 60)
        logger.info(f"【向量检索】完成，返回 {len(sections)} 个段落，总耗时={total_time:.3f}s")
        logger.info("=" * 60)

        # Top-5 日志
        for i, sec in enumerate(sections[:5]):
            heading = sec.get("heading", "")[:40] if sec.get("heading") else "无标题"
            logger.info(f"  Top-{i+1}: score={sec['score']:.4f} | {heading}...")

        return {
            "sections": sections,
            "_timings": {
                "vector_gen": vector_time,
                "vector_search": vector_search_time,
                "total": total_time,
            }
        }


__all__ = ["VectorSearcher"]
