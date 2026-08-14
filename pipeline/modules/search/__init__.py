"""
搜索模块

SAG supports VECTOR / ATOMIC / MULTI_ES / SAG2 / BM25.

架构：
- SAGSearcher/EventSearcher: 统一搜索入口（推荐使用）
- VectorSearcher: 纯向量检索器
- AtomicSearcher: 原子事项检索器
"""

from pipeline.modules.search.atomic import AtomicSearcher
from pipeline.modules.search.benchmark_utils import (
    SUPPORTED_STRATEGIES,
    activate_embedding_dim_for_source,
    build_sag2_config,
    build_strategy_config,
    create_multi_es_searcher,
    load_latest_source_info,
    normalize_section,
    search_one_question,
)
from pipeline.modules.search.bm25 import BM25ChunkSearcher
from pipeline.modules.search.config import (
    AtomicConfig,
    BM25Config,
    MultiConfig,
    RerankConfig,
    RerankStrategy,
    SAGConfig,
    SearchBaseConfig,
    SearchConfig,
    VectorConfig,
)
from pipeline.modules.search.multi_vector import ESFirstMultiSearcher
from pipeline.modules.search.searcher import (
    EventSearcher,
    SAGSearcher,
)
from pipeline.modules.search.vector import VectorSearcher

__all__ = [
    # 配置
    "SearchConfig",
    "SearchBaseConfig",
    "RerankConfig",
    "VectorConfig",
    "AtomicConfig",
    "MultiConfig",
    "SAGConfig",
    "BM25Config",
    "RerankStrategy",
    # 搜索器（推荐）
    "SAGSearcher",
    "EventSearcher",
    "VectorSearcher",
    "AtomicSearcher",
    "ESFirstMultiSearcher",
    "BM25ChunkSearcher",
    # benchmark / 脚本共享工具
    "SUPPORTED_STRATEGIES",
    "normalize_section",
    "load_latest_source_info",
    "activate_embedding_dim_for_source",
    "build_sag2_config",
    "build_strategy_config",
    "create_multi_es_searcher",
    "search_one_question",
]
