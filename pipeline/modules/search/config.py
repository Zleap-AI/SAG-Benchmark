"""Search strategy configuration.

Active strategies are VECTOR, ATOMIC, MULTI_ES, SAG2, and BM25.
SAG2 is the default graph strategy.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from pipeline.models.base import PipelineBaseModel


class RerankStrategy(str, Enum):
    """
    Public search strategy identifiers.

    MULTI_ES is the ES-first implementation. SAG2 owns graph recall,
    expansion, and LLM reranking.
    """

    VECTOR = "vector"
    ATOMIC = "atomic"
    MULTI_ES = "multi_es"
    SAG2 = "sag2"
    BM25 = "bm25"

    def __str__(self) -> str:
        return self.value


class ReturnType(str, Enum):
    """
    返回类型

    - EVENT: 事项（默认）
    - PARAGRAPH: 段落
    """

    EVENT = "event"
    PARAGRAPH = "paragraph"

    def __str__(self) -> str:
        return self.value


class RerankConfig(PipelineBaseModel):
    """
    Top-level search strategy selection. SAG2 is the default.
    """

    # 排序策略
    strategy: RerankStrategy = Field(
        default=RerankStrategy.SAG2, description="Search strategy: VECTOR/ATOMIC/MULTI_ES/SAG2/BM25"
    )


class VectorConfig(PipelineBaseModel):
    """
    向量检索器配置

    独立于三阶段搜索，直接使用 Query 向量检索 Event/Chunk，
    支持 title/heading 和 content 向量的混合搜索。

    示例：
        config = VectorConfig(
            return_type="event",
            top_k=20,
            title_weight=0.3,
            content_weight=0.7,
            similarity_threshold=0.4
        )
    """

    # 返回数量
    top_k: int = Field(default=20, ge=1, le=1000, description="最终返回的最大数量")

    # 返回类型
    return_type: Literal["chunk", "event"] = Field(
        default="event", description="返回类型：chunk=段落(SourceChunk)，event=事项(SourceEvent)"
    )

    # 向量权重（需满足 title_weight + content_weight = 1.0）
    title_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="标题向量权重（事件用 title_vector，段落用 heading_vector）",
    )

    content_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="内容向量权重（content_vector）"
    )

    # 相似度阈值（直接用作召回）
    similarity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="相似度阈值，低于此分数的结果会被过滤"
    )


class BM25Config(PipelineBaseModel):
    """
    BM25 关键词召回配置

    纯 ES multi_match BM25 检索（heading^2 + content），跳过向量/多跳/LLM 精排，
    直接返回段落。用于与向量召回、混合召回横向对比召回率。

    示例：
        config = BM25Config(top_k=20, similarity_threshold=0.0)
    """

    # 返回数量
    top_k: int = Field(default=20, ge=1, le=1000, description="最终返回的最大数量")

    # 分数阈值（ES BM25 _score 非归一化，不设上限）
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="BM25 分阈值，低于此分数的结果会被过滤（0=不过滤；注意 ES BM25 分非 [0,1]）",
    )


class QueryNormalizationConfig(PipelineBaseModel):
    """
    Query 标准化处理配置

    用于对查询进行预处理，提取关键词：
    - 文本清洗（lowercase、标点规范化）
    - jieba 分词
    - 停用词过滤（使用 goto456/stopwords 中文停用词表）
    """

    # 唯一开关
    enabled: bool = Field(
        default=False, description="是否启用 Query 标准化处理（启用后会进行分词+停用词过滤）"
    )


class SearchBaseConfig(PipelineBaseModel):
    """
    搜索基础配置

    用于引擎层统一配置，包含基础参数 + 算法配置
    """

    # 基础参数（引擎需要）
    query: str = Field(..., description="搜索查询")
    original_query: str = Field(default="", description="原始查询")
    start_time: datetime | None = Field(
        default=None,
        description="时间范围开始（可选，UTC；用于 ES 时间过滤）",
    )
    end_time: datetime | None = Field(
        default=None,
        description="时间范围结束（可选；用于时间过滤）",
    )
    source_ids: list[str] | None = Field(
        default=None,
        description="事项来源ID列表（Article/Conversation ID），可选，用于精确过滤",
    )

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def _strip_timezone(cls, v):
        """
        数据库存储 UTC 时间（MySQL connection SET time_zone='+00:00'）。
        前端传入 ISO 8601 带时区偏移（本地时间），需转为 naive UTC 匹配 DB。
        """
        if v is not None and v.tzinfo is not None:
            return v.astimezone(UTC).replace(tzinfo=None)
        return v

    # 功能开关
    enable_query_rewrite: bool = Field(
        default=True, description="启用query重写（将口语化表述整理为更适合查询的问题）"
    )

    # Query 标准化配置
    query_normalization: QueryNormalizationConfig = Field(
        default_factory=QueryNormalizationConfig, description="Query 标准化处理配置"
    )

    # 实体类型过滤（Recall 和 Expand 阶段都使用）
    exclude_entity_types: list[str] = Field(
        default=["start_time", "end_time"], description="[黑名单] 需要排除的实体类型"
    )

    # 返回类型控制
    return_type: ReturnType = Field(
        default=ReturnType.EVENT, description="返回类型：事项(event) 或 段落(paragraph)，默认是事项"
    )

    # 线索数量控制（统一控制 Expand 和 Rerank 阶段）
    max_clues_per_event: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Expand阶段：限制事项-实体双向线索；Rerank阶段：限制事项的实体线索",
    )

    # 段落返回控制
    return_chunks: bool = Field(
        default=False, description="是否返回段落信息（从事项的chunk_id获取，按事项排序去重）"
    )

    # 重排配置
    rerank: RerankConfig = Field(default_factory=RerankConfig, description="重排配置")

    # 策略专属配置（透传给 SearchConfig.strategy_config）
    strategy_config: Any | None = Field(
        default=None, description="策略专属配置实例（MultiConfig/AtomicConfig/VectorConfig）"
    )

    # 证据追踪（SAG2 用，传入 gold evidence ID 列表后各阶段打印覆盖率）
    gold_evidences: list[str] | None = Field(
        default=None,
        description="该问题的 gold evidence ID 列表（SAG2 EvidenceTracker 使用，None=不追踪）",
    )


class SearchConfig(SearchBaseConfig):
    """
    搜索完整配置（基础配置 + 运行时上下文）

    继承SearchBaseConfig，添加运行时必需的上下文信息

    示例：
        # 单源搜索（向后兼容）
        config = SearchConfig(
            query="人工智能",
            source_config_id="source_123",
            recall=RecallConfig(max_entities=30),
            expand=ExpandConfig(max_hops=3),
            rerank=RerankConfig(strategy=RerankStrategy.SAG2)
        )

        # 多源搜索（新增功能）
        config = SearchConfig(
            query="人工智能",
            source_config_ids=["source_001", "source_002", "source_003"],
            recall=RecallConfig(max_entities=30),
            expand=ExpandConfig(max_hops=3),
            rerank=RerankConfig(strategy=RerankStrategy.SAG2)
        )
    """

    # === 运行时上下文 ===
    source_config_id: str | None = Field(None, description="数据源ID（单个，向后兼容）")
    source_config_ids: list[str] | None = Field(None, description="数据源ID列表（支持多源搜索）")
    article_id: str | None = Field(None, description="文章ID")
    background: str | None = Field(None, description="背景信息")

    # === 策略专属配置（可选，传入后由 SAGSearcher 透传给对应子 searcher）===
    strategy_config: Any | None = Field(
        default=None,
        description="策略专属配置实例（MultiConfig/AtomicConfig/VectorConfig），"
        "传入后 SAGSearcher 会将其透传给对应子 searcher，覆盖子 searcher 的默认值",
    )

    def model_post_init(self, __context):
        """初始化后验证和处理 source_config_id/source_config_ids"""
        # 验证：至少提供一个
        if not self.source_config_id and not self.source_config_ids:
            raise ValueError("必须提供 source_config_id 或 source_config_ids 参数")

        # 统一处理：如果只提供 source_config_id，转换为 source_config_ids
        if self.source_config_id and not self.source_config_ids:
            self.source_config_ids = [self.source_config_id]
        elif self.source_config_ids and not self.source_config_id:
            # 多源场景，source_config_id 设为第一个（向后兼容）
            self.source_config_id = self.source_config_ids[0]

    def get_source_config_ids(self) -> list[str]:
        """
        获取统一的 source_config_ids 列表

        Returns:
            source_config_ids 列表（至少包含一个元素）
        """
        return self.source_config_ids or []

    def is_multi_source(self) -> bool:
        """是否为多源搜索"""
        return len(self.get_source_config_ids()) > 1


class AtomicConfig(PipelineBaseModel):
    """
    原子事项检索器配置

    检索恰好包含 2 个实体的原子化三元组事项。

    示例：
        config = AtomicConfig(
            top_k=20,
            similarity_threshold=0.4
        )
    """

    entity_top_k: int = Field(default=20, ge=1, le=1000, description="返回的最大实体数量")
    atomic_top_k: int = Field(default=20, ge=1, le=1000, description="原子化事项最大数量")

    key_similarity_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="相似度阈值，低于此分数的结果会被过滤"
    )

    similarity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="事项向量检索相似度阈值"
    )

    max_hops: int = Field(default=1, ge=0, le=10, description="多跳扩展次数（0=不扩展，1=扩展1轮）")

    max_events: int = Field(default=1000, ge=1, le=5000, description="粗排序最大返回事项数量")

    rerank_top_k: int = Field(default=5, ge=1, le=20, description="LLM 精选返回数量")

    max_sections: int = Field(
        default=10, ge=1, le=50, description="最终返回段落最大数量（chunk_id 去重后截断）"
    )


class SAG2RecallConfig(PipelineBaseModel):
    """SAG2 召回配置"""

    max_entities: int = Field(
        default=15, ge=1, le=100, description="query→entity BM25/向量召回数量"
    )
    query_recall_event_max: int = Field(
        default=20, ge=1, le=200, description="路A query→event 保留上限"
    )
    max_events_per_key: int = Field(
        default=10, ge=1, le=100, description="每个实体读取的关系数上限"
    )
    entity_vector_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="LOCAL 实体向量匹配阈值"
    )
    score_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Path A/B 相似度过滤阈值"
    )


class SAG2ExpandConfig(PipelineBaseModel):
    """SAG2 多跳扩展配置"""

    enabled: bool = Field(default=True, description="是否启用扩展")
    max_hops: int = Field(default=1, ge=0, le=2, description="扩展跳数")
    entities_per_hop: int = Field(default=15, ge=1, le=100, description="每跳新增实体上限")
    max_events_per_hop: int = Field(
        default=50, ge=1, le=500, description="每跳新增事项上限"
    )
    event_similarity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="expand 中 entity→event 相似度阈值"
    )
    entity_relation_score_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="expand 中 event→new key 关系向量阈值（ES kNN cosine 过滤）",
    )
    seed_event_limit: int = Field(default=15, ge=1, le=100, description="种子 event 取 top N")
    relation_k_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="event→new key 关系打分候选放大倍率（relation_k = entities_per_hop × 此值）",
    )


class SAG2ScopeConfig(PipelineBaseModel):
    # Optional bounded SAG2 candidate-pool and in-memory subgraph scope.

    enabled: bool = Field(
        default=False,
        description="Build a bounded event/entity universe before SAG2 recall.",
    )
    event_top_k: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Number of top query-similar events used to bootstrap the scope.",
    )
    bootstrap_entity_limit: int = Field(
        default=0,
        ge=0,
        le=1000000,
        description="Optional cap on event-entity relations; 0 means unlimited.",
    )
    include_event_content: bool = Field(
        default=True,
        description="Keep event content in memory for ranking; chunks remain SQL hydrated.",
    )


class SAG2RerankConfig(PipelineBaseModel):
    """SAG2 排序配置"""

    strategy: Literal["rerank", "llm_rank", "rrf"] = Field(
        default="llm_rank", description="排序策略"
    )
    score_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="候选 event 向量相似度阈值"
    )
    rerank_score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="rerank 模型分阈值"
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="rerank 模型真正保留的结果数；不足 max_results 时按 embedding 相似度补齐",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF 名次融合常数")
    max_results: int = Field(default=10, ge=1, le=100, description="最终返回事项数")
    rerank_timeout: float = Field(
        default=60.0, ge=1.0, le=60.0, description="rerank 超时秒数"
    )
    llm_rank_top_n: int = Field(default=100, ge=1, le=200, description="送 LLM 前粗排上限")
    llm_rank_max_results: int = Field(
        default=5, ge=1, le=20, description="LLM 排序返回上限"
    )
    llm_rank_include_content: bool = Field(default=True, description="LLM 排序是否带全文")
    llm_rank_max_content_len: int = Field(
        default=2000, ge=100, description="每条 content 截断字符数"
    )
    llm_rank_prompt: str = Field(
        default="llm_rank_events",
        description="LLM 排序模板名（select_useful_relations_local=KG 关系选择 / llm_rank_events=index+score）",
    )


class SAGConfig(PipelineBaseModel):
    """Independent SAG2 configuration — decoupled from MultiConfig.

    SAG2 reads only this config. MultiES continues to read MultiConfig.
    The two config classes share no inheritance relationship.
    """

    strategy: Literal["sag2"] = "sag2"

    use_mlflow_prompts: bool = Field(
        default=False,
        description="是否从 MLflow Prompt Registry 加载提示词（启动时一次性加载并打印来源；"
        "未注册的条目回退到代码常量，服务不可达则报错）。默认 False=全用代码常量。",
    )
    mlflow_prompt_alias: str = Field(
        default="latest",
        description="MLflow Prompt Registry 别名（use_mlflow_prompts=True 时生效）。"
        "加载时使用 prompts:/name@{alias}，默认为 latest。"
        "可设为 production/staging 等具名别名实现版本切换。",
    )
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="MLflow Tracking URI（use_mlflow_prompts=True 时透传给 PromptProvider，"
        "用于确保 prompt 加载使用正确的 HTTP endpoint）。"
        "传入后 PromptProvider.load_all() 会调用 mlflow.set_tracking_uri(该值)。",
    )

    max_sections: int = Field(
        default=10, ge=1, le=50, description="最终返回段落最大数量（chunk_id 去重后截断）"
    )

    sag2_recall: SAG2RecallConfig = Field(
        default_factory=SAG2RecallConfig, description="SAG2 召回配置"
    )
    sag2_scope: SAG2ScopeConfig = Field(default_factory=SAG2ScopeConfig)
    sag2_expand: SAG2ExpandConfig = Field(
        default_factory=SAG2ExpandConfig, description="SAG2 多跳扩展配置"
    )
    sag2_rerank: SAG2RerankConfig = Field(
        default_factory=SAG2RerankConfig, description="SAG2 排序配置"
    )

    sag2_rewrite_query_enabled: bool = Field(
        default=False, description="SAG2 LLM 问题重写开关"
    )
    sag2_enable_entity_extraction: bool = Field(
        default=True, description="SAG2 独立 NER 开关（默认开启，对齐 benchmark --foundation search）"
    )
    sag2_use_fast_mode: bool = Field(
        default=False, description="SAG2 快速模式（跳过 LLM 排序）"
    )


class MultiConfig(PipelineBaseModel):
    """
    Configuration for the MULTI_ES search route.

    SAG2 has its own ``SAGConfig``. The ``sag2_*`` fields below remain only for
    migration of older callers and are not read by the MULTI_ES implementation.
    """

    # ========== 策略选择 ==========
    strategy: str = Field(
        default="multi_es",
        description="MULTI_ES strategy identifier",
    )
    mode: Literal["fast", "precise"] = Field(
        default="fast",
        description="multi_vector 模式：fast=BM25实体召回+小规模扩展，precise=BM25实体召回+LLM过滤",
    )
    # ========== 提示词来源（MLflow Prompt Registry）==========
    use_mlflow_prompts: bool = Field(
        default=False,
        description="是否从 MLflow Prompt Registry 加载提示词（启动时一次性加载并打印来源；"
        "未注册的条目回退到代码常量，服务不可达则报错）。默认 False=全用代码常量。",
    )
    mlflow_prompt_alias: str = Field(
        default="latest",
        description="MLflow Prompt Registry 别名（use_mlflow_prompts=True 时生效）。"
        "加载时使用 prompts:/name@{alias}，默认为 latest。"
        "可设为 production/staging 等具名别名实现版本切换。",
    )
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="MLflow Tracking URI（use_mlflow_prompts=True 时透传给 PromptProvider，"
        "用于确保 prompt 加载使用正确的 HTTP endpoint）。"
        "传入后 PromptProvider.load_all() 会调用 mlflow.set_tracking_uri(该值)。",
    )

    # ========== 通用参数 ==========
    entity_top_k: int = Field(
        default=20, ge=1, le=1000, description="query->entity 返回的最大实体数量（MULTI_ES 使用 BM25 召回）"
    )
    multi_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="query->event 向量召回数量；precise 的 entity->event 通道在 multi_vector 中固定使用 40",
    )
    similarity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="事项向量检索相似度阈值"
    )
    max_sections: int = Field(
        default=10, ge=1, le=50, description="最终返回段落最大数量（chunk_id 去重后截断）"
    )
    # ========== MULTI_ES 策略专用参数 ==========
    max_hops: int = Field(
        default=1, ge=0, le=10, description="[multi_es] 多跳扩展次数（0=不扩展，1=扩展1轮）"
    )
    max_events: int = Field(
        default=100, ge=1, le=500, description="[multi_es] 粗排序最大返回事项数量")
    max_expand_events_per_hop: int = Field(
        default=2000,
        ge=1,
        le=10000,
        description="[multi_es] 扩展阶段每跳 entity->event 最多召回事项数量",
    )

    fast_entity_k: int = Field(
        default=5, ge=1, le=100, description="[fast] key->entity 后保留的实体数量"
    )
    fast_entity_event_candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="[fast] entity->event 初始候选数量，默认限制为 20，随后按 query-content 相似度取 fast_entity_event_k",
    )
    fast_entity_event_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="[fast] entity 过滤候选内按 query-content 相似度保留的 event1 数量",
    )
    fast_query_event_k: int = Field(
        default=20, ge=1, le=100, description="[fast] query->event2 直接向量召回数量"
    )
    fast_answer_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="[fast] event1/event2 并集后按分数保留的第一跳 event 数量",
    )
    fast_expand_answer_k: int = Field(
        default=5,
        ge=0,
        le=100,
        description="[fast] 第一跳扩展后的 event_set2 按 query-content 相似度保留数量",
    )
    fast_vector_weight: float = Field(
        default=0.85,
        ge=0.0,
        description="[fast] expand 前第一跳事项排序的 query-content 向量分权重",
    )
    fast_entity_weight: float = Field(
        default=0.15,
        ge=0.0,
        description="[fast] expand 前第一跳事项排序的实体命中增强权重；命中任一 fast entity 记 1，否则记 0",
    )
    fast_channel_weight: float = Field(
        default=0.05, ge=0.0, description="[fast] expand 前第一跳事项排序的双通道命中奖励权重"
    )

    # ========== SAG2 专用配置（SAG2Searcher 读取，其它策略忽略；向后兼容）==========
    sag2_recall: SAG2RecallConfig = Field(
        default_factory=SAG2RecallConfig, description="SAG2 召回配置"
    )
    sag2_scope: SAG2ScopeConfig = Field(default_factory=SAG2ScopeConfig)
    sag2_expand: SAG2ExpandConfig = Field(
        default_factory=SAG2ExpandConfig, description="SAG2 多跳扩展配置"
    )
    sag2_rerank: SAG2RerankConfig = Field(
        default_factory=SAG2RerankConfig, description="SAG2 排序配置"
    )
    sag2_rewrite_query_enabled: bool = Field(
        default=False, description="SAG2 LLM 问题重写开关"
    )
    sag2_enable_entity_extraction: bool = Field(
        default=True, description="SAG2 独立 NER 开关（默认开启，对齐 benchmark --foundation search）"
    )
    sag2_use_fast_mode: bool = Field(
        default=False, description="SAG2 快速模式（跳过 LLM 排序）"
    )

__all__ = [
    # 配置
    "SearchConfig",
    "SearchBaseConfig",
    "RerankConfig",
    "VectorConfig",
    "AtomicConfig",
    "MultiConfig",
    "SAGConfig",
    "SAG2RecallConfig",
    "SAG2ExpandConfig",
    "SAG2ScopeConfig",
    "SAG2RerankConfig",
    "BM25Config",
    "QueryNormalizationConfig",
    "RerankStrategy",
    "ReturnType",
]
