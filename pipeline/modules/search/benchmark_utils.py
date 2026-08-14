"""
搜索脚本 / benchmark 共享工具

供 scripts/run_search.py、scripts/run_search_benchmark.py、scripts/run_qa_benchmark.py
复用，统一 section 归一化、source_config_id 解析和策略配置构造，
避免 recall / QA 计分因入口脚本差异而不可比。
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.modules.search.config import SAGConfig

# 支持的检索策略（与 benchmark 一致）
SUPPORTED_STRATEGIES = [
    "atomic",
    "multi_es",
    "sag2",
    "vector",
    "bm25",
]


def normalize_section(s) -> str:
    """将一个 section（dict 或 str）归一化为 "heading\\ncontent" 字符串。

    与 gold 文档格式（load_utils 中的 "title\\ncontent"）对齐：**不加 `#` 前缀**，
    并剥离 content 首行与 heading 重复的 markdown 标题行。recall 精确匹配依赖此格式，
    任何前缀/格式差异都会导致召回归零。
    """
    if isinstance(s, str):
        return s
    heading = s.get("heading", "") or s.get("title", "") or ""
    content = s.get("content", "") or ""
    # 去掉 content 开头的 markdown 标题行（"# ..." 格式）
    lines = content.split("\n")
    if lines and lines[0].strip().lstrip("#").strip() == heading.strip():
        content = "\n".join(lines[1:]).lstrip("\n")
    return f"{heading}\n{content}"


def load_latest_source_info(dataset_name: str, model_name: str) -> dict[str, Any]:
    """
    从 pipeline/evaluation/source/SAG/{model_name}/{dataset_name}/{timestamp}/
    加载指定模型下最新时间戳文件夹的 source_info.json。

    Args:
        dataset_name: 数据集名称
        model_name: 模型名称（从 .env 的 LLM_MODEL 读取）

    Returns:
        包含 source_config_id, model_name, timestamp 等信息的字典

    Raises:
        FileNotFoundError: 模型目录、数据集目录或 source_info.json 不存在
    """
    # 本文件位于 pipeline/modules/search/ 下，项目根 = 向上回退三级
    sag_base_dir = (
        Path(__file__).parent.parent.parent.parent / "pipeline" / "evaluation" / "source" / "SAG"
    )

    if not sag_base_dir.exists():
        raise FileNotFoundError(f"SAG directory not found: {sag_base_dir}")

    # 直接定位到指定模型的目录
    model_dir = sag_base_dir / model_name
    if not model_dir.exists():
        available_models = [d.name for d in sag_base_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"模型目录不存在: {model_dir}\n"
            f"可用模型: {available_models}\n"
            f"提示: 请检查 .env 文件中的 LLM_MODEL 配置是否正确"
        )

    dataset_dir = model_dir / dataset_name
    if not dataset_dir.exists():
        available_datasets = [d.name for d in model_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"数据集目录不存在: {dataset_dir}\n"
            f"模型 {model_name} 下可用的数据集: {available_datasets}"
        )

    # 收集该模型该数据集下所有时间戳的 source_info.json
    all_source_info_files = []
    for ts_dir in dataset_dir.iterdir():
        if ts_dir.is_dir():
            source_info_path = ts_dir / "source_info.json"
            if source_info_path.exists():
                all_source_info_files.append(source_info_path)

    if not all_source_info_files:
        raise FileNotFoundError(f"在 {dataset_dir} 下未找到任何 source_info.json 文件")

    # 按时间戳排序（目录名格式：YYYYMMDD_HHMMSS），选择最新的
    latest_source_info_path = max(all_source_info_files, key=lambda p: p.parent.name)

    with open(latest_source_info_path, encoding="utf-8") as f:
        info = json.load(f)

    return {
        "source_config_id": info.get("source_config_id"),
        "dataset_name": info.get("dataset_name"),
        "timestamp": info.get("timestamp"),
        "source_name": info.get("source_name"),
        "file_path": str(latest_source_info_path),
        # embedding 元数据为可选字段；缺失时由维度激活逻辑处理。
        "embedding": info.get("embedding"),
    }


async def activate_embedding_dim_for_source(
    source_info: dict[str, Any] | None,
    *,
    allow_mismatch: bool = False,
) -> int:
    """根据 source_info 的 embedding 元数据设定检索使用的向量维度。

    优先级：source_info.embedding.dim（该 run 落盘时的真实维度）
          > 当前环境 probe 结果
          > LEGACY_DIM（老 source_info 无 embedding 块时）

    维度决定读哪个物理索引；source_config_id 仍负责 run 级过滤（双层隔离）。
    """
    from pipeline.core.ai.embedding_dim import resolve_embedding_dim
    from pipeline.storage.indexing import (
        LEGACY_DIM,
        resolve_all_index_names,
        set_active_embedding_dim,
    )
    from pipeline.utils import get_logger

    logger = get_logger("search.benchmark_utils")

    emb = (source_info or {}).get("embedding") or {}
    source_dim = emb.get("dim")

    try:
        current = await resolve_embedding_dim()
        current_dim = current["dim"]
    except Exception as e:  # noqa: BLE001 —— 检索可在离线索引上跑，probe 失败不该致命
        logger.warning(f"当前环境 embedding 维度探测失败: {e}")
        current_dim = None

    if source_dim is None:
        dim = current_dim or LEGACY_DIM
        logger.warning(
            f"⚠️  source_info.json 无 embedding 元数据（旧版 upload 产物），"
            f"按 dim={dim} 处理。若召回异常请用新版 run_upload.py 重新上传。"
        )
    else:
        dim = source_dim
        if current_dim is not None and current_dim != source_dim:
            msg = (
                f"❌ embedding 维度不一致：该 run 建库时 dim={source_dim} "
                f"(model={emb.get('model')})，当前 .env 端点 dim={current_dim}。\n"
                f"   用不同 embedding 模型的向量做召回会污染 benchmark 结果。\n"
                f"   如确认只是想复用旧索引，请加 --allow-embedding-mismatch。"
            )
            if not allow_mismatch:
                raise ValueError(msg)
            logger.error(msg + " —— 已按 --allow-embedding-mismatch 继续（结果不可比）")

    set_active_embedding_dim(dim)
    logger.info(f"🧭 检索使用 embedding 维度: {dim}，索引: {resolve_all_index_names(dim)}")
    return dim


def build_sag2_config(
    *,
    max_sections: int,
    rerank_top_k: int | None = None,
    max_results: int | None = None,
    scope_enabled: bool = False,
    event_top_k: int = 1000,
    bootstrap_entity_limit: int = 0,
    include_event_content: bool = True,
) -> "SAGConfig":
    """构造 SAG2 配置，并校验最终返回数量与重排数量的关系。"""
    from pipeline.modules.search.config import SAG2RerankConfig, SAGConfig

    rerank_overrides: dict[str, int] = {}
    if rerank_top_k is not None:
        rerank_overrides["rerank_top_k"] = rerank_top_k
    if max_results is not None:
        rerank_overrides["max_results"] = max_results

    config = SAGConfig(
        max_sections=max_sections,
        sag2_scope={
            "enabled": scope_enabled,
            "event_top_k": event_top_k,
            "bootstrap_entity_limit": bootstrap_entity_limit,
            "include_event_content": include_event_content,
        },
        sag2_rerank=SAG2RerankConfig(**rerank_overrides),
    )
    if config.sag2_rerank.rerank_top_k > config.sag2_rerank.max_results:
        raise ValueError("sag2_rerank_top_k 不能大于 sag2_max_results")
    return config


def build_strategy_config(
    strategy: str,
    top_k: int,
    mode: str = "fast",
) -> tuple[Any, Any]:
    """根据策略字符串构造 ``(RerankStrategy, strategy_config)``。"""
    from pipeline.modules.search.config import (
        AtomicConfig,
        BM25Config,
        MultiConfig,
        RerankStrategy,
        SAGConfig,
        VectorConfig,
    )

    strategy_config_map = {
        "atomic": (RerankStrategy.ATOMIC, lambda: AtomicConfig(max_sections=top_k)),
        "multi_es": (
            RerankStrategy.MULTI_ES,
            lambda: MultiConfig(strategy="multi_es", mode=mode, max_sections=top_k),
        ),
        "sag2": (
            RerankStrategy.SAG2,
            lambda: SAGConfig(max_sections=top_k),
        ),
        "vector": (RerankStrategy.VECTOR, lambda: VectorConfig(top_k=top_k)),
        "bm25": (RerankStrategy.BM25, lambda: BM25Config(top_k=top_k)),
    }

    if strategy not in strategy_config_map:
        raise ValueError(f"不支持的策略: {strategy}（可选: {', '.join(strategy_config_map)}）")

    search_strategy, config_factory = strategy_config_map[strategy]
    return search_strategy, config_factory()


async def create_multi_es_searcher(strategy_config) -> Any:
    """为 multi_es 策略创建并预热 MultiSearcherES。返回已 warmup 的 searcher。"""
    from pipeline.modules.search.multi_vector import MultiSearcherES as ESMultiSearcher

    searcher = ESMultiSearcher(config=strategy_config)
    await searcher.warmup(strategy_config)
    return searcher


async def search_one_question(
    question: str,
    source_config_id: str,
    search_strategy,
    strategy_config,
    top_k: int,
    multi_es_searcher=None,
) -> list[str]:
    """对单个问题执行检索，返回归一化后的 section 字符串列表（已截断 top_k）。

    与 benchmark 入口复用相同的策略配置与 section 归一化规则：
    - multi_es 策略走预热好的 MultiSearcherES.search_for_sections
    - 其余策略统一通过 PipelineEngine 执行

    engine 的池化与 benchmark 统计采集由批量入口自行管理，不属于本函数职责。
    """
    from pipeline import PipelineEngine
    from pipeline.engine.config import TaskConfig
    from pipeline.modules.search.config import RerankConfig, ReturnType, SearchBaseConfig

    if multi_es_searcher is not None:
        raw = await multi_es_searcher.search_for_sections(
            query=question,
            source_config_ids=[source_config_id],
            config=strategy_config,
        )
        raw_sections = raw.get("sections", [])
        return [normalize_section(s) for s in raw_sections][:top_k]

    engine = PipelineEngine(
        task_config=TaskConfig(
            task_name="search",
            source_config_id=source_config_id,
        ),
        auto_setup_logging=False,
    )
    await engine.search_async(
        SearchBaseConfig(
            query=question,
            return_type=ReturnType.PARAGRAPH,
            rerank=RerankConfig(strategy=search_strategy),
            strategy_config=strategy_config,
        )
    )
    engine_result = engine.get_result()
    raw_sections = (
        engine_result.search_result.data_full
        if engine_result and engine_result.search_result
        else []
    )
    return [normalize_section(s) for s in raw_sections][:top_k]
