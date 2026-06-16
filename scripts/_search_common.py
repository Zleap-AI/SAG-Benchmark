#!/usr/bin/env python3
"""
搜索脚本共享工具

供 scripts/run_search.py 与 scripts/run_search_benchmark.py 复用，确保两者在
section 归一化、source_config_id 解析、策略调度上行为完全一致，避免逻辑漂移导致
recall 计分不可比。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# 支持的检索策略（与 benchmark 一致）
SUPPORTED_STRATEGIES = ["atomic", "multi", "multi_es", "multi1", "hopllm", "vector"]


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


def load_latest_source_info(dataset_name: str, model_name: str) -> Dict[str, Any]:
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
    # 本文件位于 scripts/ 下，项目根 = 上一级
    sag_base_dir = Path(__file__).parent.parent / "pipeline" / "evaluation" / "source" / "SAG"

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
        raise FileNotFoundError(
            f"在 {dataset_dir} 下未找到任何 source_info.json 文件"
        )

    # 按时间戳排序（目录名格式：YYYYMMDD_HHMMSS），选择最新的
    latest_source_info_path = max(all_source_info_files, key=lambda p: p.parent.name)

    with open(latest_source_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return {
        "source_config_id": info.get("source_config_id"),
        "dataset_name": info.get("dataset_name"),
        "timestamp": info.get("timestamp"),
        "source_name": info.get("source_name"),
        "file_path": str(latest_source_info_path),
    }


def build_strategy_config(strategy: str, top_k: int, mode: str = "fast"):
    """根据策略字符串构造 (RerankStrategy, strategy_config)。

    与 run_search_benchmark.py 的 strategy_config_map 保持一致。
    """
    from pipeline.modules.search.config import (
        RerankStrategy,
        MultiConfig,
        AtomicConfig,
        VectorConfig,
    )

    strategy_config_map = {
        "atomic": (RerankStrategy.ATOMIC, lambda: AtomicConfig(max_sections=top_k)),
        "multi": (RerankStrategy.MULTI, lambda: MultiConfig(strategy="multi", max_sections=top_k)),
        "multi_es": (
            RerankStrategy.MULTI_ES,
            lambda: MultiConfig(strategy="multi", mode=mode, max_sections=top_k),
        ),
        "multi1": (RerankStrategy.MULTI, lambda: MultiConfig(strategy="multi1", max_sections=top_k)),
        "hopllm": (RerankStrategy.MULTI, lambda: MultiConfig(strategy="hopllm", max_sections=top_k)),
        "vector": (RerankStrategy.VECTOR, lambda: VectorConfig(top_k=top_k)),
    }

    if strategy not in strategy_config_map:
        raise ValueError(
            f"不支持的策略: {strategy}（可选: {', '.join(strategy_config_map)}）"
        )

    search_strategy, config_factory = strategy_config_map[strategy]
    return search_strategy, config_factory()


async def create_multi_es_searcher(strategy_config):
    """为 multi_es 策略创建并预热 MultiSearcherES。返回已 warmup 的 searcher。"""
    from pipeline.modules.search.multi_vector import MultiSearcher as ESMultiSearcher

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
) -> List[str]:
    """对单个问题执行检索，返回归一化后的 section 字符串列表（已截断 top_k）。

    调度逻辑与 run_search_benchmark.run_batch_search 完全一致：
    - multi_es 策略走预热好的 MultiSearcherES.search_for_sections
    - 其余策略统一通过 pipelineEngine 执行
    """
    from pipeline import pipelineEngine
    from pipeline.engine.config import TaskConfig
    from pipeline.modules.search.config import (
        SearchBaseConfig,
        RerankConfig,
        ReturnType,
    )

    if multi_es_searcher is not None:
        raw = await multi_es_searcher.search_for_sections(
            query=question,
            source_config_ids=[source_config_id],
            config=strategy_config,
        )
        raw_sections = raw.get("sections", [])
        return [normalize_section(s) for s in raw_sections][:top_k]

    engine = pipelineEngine(
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
