#!/usr/bin/env python3
"""
搜索 + Benchmark 脚本

展示逻辑和计分逻辑与 pipeline/evaluation/benchmark.py 完全一致。

详细参数说明和使用示例见 docs/search.md。
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from pipeline.core.config import get_settings
from pipeline.evaluation.metrics import RetrievalRecall
from pipeline.evaluation.utils import (
    DatasetLoader,
    LLMTokenTracker,
    MLflowConfig,
    MLflowTracker,
    get_local_ip,
    llm_tracking_scope,
    llm_tracking_stage,
)
from pipeline.evaluation.utils.step_timing import (
    _build_timing_mlflow_metrics,
    _build_token_mlflow_metrics,
    _has_step_timing,
    batch_view_from_cumulative,
    build_supplementary_metrics,
    compute_step_timings,
    public_search_diagnostics,
    raw_diagnostic_snapshot,
)
from pipeline.modules.search import (
    build_sag2_config,
    load_latest_source_info,
    normalize_section,
)
from pipeline.utils import get_logger

logger = get_logger("scripts.run_search_benchmark")

# 压制 pipeline 内部的详细日志，只保留 WARNING 以上
# 注意：项目内 get_logger(name) 会自动加 "pipeline." 前缀（见 pipeline/utils/logger.py），
# 因此实际 logger 名是 "pipeline.search.*" / "pipeline.ai.*"。
logging.getLogger("pipeline").setLevel(logging.WARNING)
# 放开搜索配置块（pipeline.search.sag / pipeline.search.searcher）与各策略检索流程日志
logging.getLogger("pipeline.search.sag").setLevel(logging.INFO)
logging.getLogger("pipeline.search.searcher").setLevel(logging.INFO)
logging.getLogger("pipeline.search.multi_es").setLevel(logging.INFO)
logging.getLogger("pipeline.search.sag2").setLevel(logging.INFO)
logging.getLogger("pipeline.search.vector").setLevel(logging.INFO)
logging.getLogger("pipeline.search.atomic").setLevel(logging.INFO)
# 放开提示词加载日志（pipeline.search.prompts）：启动时一次性打印
# 「当前 LLM 实际使用的每条提示词全文 + 来源/版本」，便于确认 MLflow 版本内容
logging.getLogger("pipeline.search.prompts").setLevel(logging.INFO)
# 放开 LLM 重试日志，方便观察重试次数和等待时间
logging.getLogger("pipeline.ai.llm").setLevel(logging.INFO)
# 放开 LLM 调用客户端日志到 DEBUG，显示完整的请求消息 / 响应内容
# （logger 名 ai.openai → pipeline.ai.openai，见 pipeline/core/ai/llm.py）
# 注意：DEBUG 内容能否真正输出还取决于 handler 级别，见 main() 中 console/file handler 设置
logging.getLogger("pipeline.ai.openai").setLevel(logging.DEBUG)

# 压制第三方库的 DEBUG 噪声：basicConfig(level=DEBUG) 会让根级别放行 DEBUG，
# 但 httpx/httpcore/openai/elasticsearch 等库的 DEBUG 极其啰嗦（每个 HTTP
# 请求的 headers/body），且它们不在 pipeline 命名空间下、不受上面 WARNING 约束，
# 因此在此显式压到 WARNING，只保留我们关心的 pipeline.ai.openai 完整请求/响应。
for _noisy in ("httpx", "httpcore", "openai", "elasticsearch", "urllib3", "asyncio", "aiomysql"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def calculate_precision_f1_at_k(
    gold_docs: list[list[str]],
    retrieved_docs: list[list[str]],
    k_list: list[int],
) -> dict[str, float]:
    """
    计算 Precision@K 和 F1@K。
    与 benchmark.py 的 Evaluate._calculate_precision_f1_at_k 完全一致。
    """
    k_list = sorted(set(k_list))
    pooled_precision = dict.fromkeys(k_list, 0.0)
    pooled_f1 = dict.fromkeys(k_list, 0.0)

    num_examples = len(gold_docs)
    if num_examples == 0:
        result: dict[str, float] = {}
        for k in k_list:
            result[f"Precision@{k}"] = 0.0
            result[f"F1@{k}"] = 0.0
        return result

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        gold_set = set(example_gold_docs)
        for k in k_list:
            top_k_docs = example_retrieved_docs[:k]
            relevant_retrieved = set(top_k_docs) & gold_set

            precision = len(relevant_retrieved) / len(top_k_docs) if top_k_docs else 0.0
            recall = len(relevant_retrieved) / len(gold_set) if gold_set else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            )

            pooled_precision[k] += precision
            pooled_f1[k] += f1

    pooled_results: dict[str, float] = {}
    for k in k_list:
        pooled_results[f"Precision@{k}"] = round(pooled_precision[k] / num_examples, 4)
        pooled_results[f"F1@{k}"] = round(pooled_f1[k] / num_examples, 4)
    return pooled_results


async def run_batch_search(
    questions: list[str],
    sample_ids: list[str],
    source_config_id: str,
    strategy: str,
    mode: str,
    top_k: int,
    max_concurrency: int,
    bench_size: int,
    gold_docs_for_recall: list[list[str]],
    mlflow_tracker: Any | None,
    bench_logger,
    use_mlflow_prompts: bool = False,
    mlflow_prompt_alias: str = "latest",
    mlflow_url: str | None = None,
    sag2_scope_enabled: bool = False,
    sag2_event_top_k: int = 1000,
    sag2_bootstrap_entity_limit: int = 0,
    sag2_include_event_content: bool = True,
    sag2_rerank_top_k: int | None = None,
    sag2_max_results: int | None = None,
) -> list[dict]:
    """
    批量搜索并按 bench_size 触发进度回调。

    engine 路径使用并发隔离的实例池；独立直连策略沿用其自身执行入口。

    返回格式与 benchmark.py 的 search_results 完全一致：
    [
        {
            'question_index': int,
            'question': str,
            'sections': List[str],       # "title\ncontent" 格式
        },
        ...
    ]
    """
    from pipeline import PipelineEngine
    from pipeline.engine.config import TaskConfig
    from pipeline.modules.search.config import (
        AtomicConfig,
        BM25Config,
        MultiConfig,
        RerankConfig,
        RerankStrategy,
        ReturnType,
        SAGConfig,
        SearchBaseConfig,
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
            lambda: build_sag2_config(
                max_sections=top_k,
                rerank_top_k=sag2_rerank_top_k,
                max_results=sag2_max_results,
                scope_enabled=sag2_scope_enabled,
                event_top_k=sag2_event_top_k,
                bootstrap_entity_limit=sag2_bootstrap_entity_limit,
                include_event_content=sag2_include_event_content,
            ),
        ),
        "vector": (RerankStrategy.VECTOR, lambda: VectorConfig(top_k=top_k)),
        "bm25": (RerankStrategy.BM25, lambda: BM25Config(top_k=top_k)),
    }

    search_strategy, config_factory = strategy_config_map[strategy]
    strategy_config = config_factory()

    # 提示词来源：若启用 MLflow Prompt Registry，把开关注入到当前策略配置。
    # 显式传入 mlflow_url，绕过全局 _tracking_uri 状态依赖，
    # PromptProvider.load_all() 会用该 URI 调用 mlflow.set_tracking_uri。
    if use_mlflow_prompts and isinstance(strategy_config, MultiConfig | SAGConfig):
        strategy_config.use_mlflow_prompts = True
        strategy_config.mlflow_prompt_alias = mlflow_prompt_alias
        if mlflow_url:
            strategy_config.mlflow_tracking_uri = mlflow_url
        bench_logger.info(
            "[Prompt] 启用 MLflow Prompt Registry: alias=%s, uri=%s",
            mlflow_prompt_alias,
            mlflow_url or "(未指定，依赖全局 tracking_uri)",
        )

    multi_es_searcher = None
    if strategy == "multi_es":
        from pipeline.modules.search.multi_vector import MultiSearcherES as ESMultiSearcher

        multi_es_searcher = ESMultiSearcher(config=strategy_config)
        await multi_es_searcher.warmup(strategy_config)

    if len(sample_ids) != len(questions):
        raise ValueError("sample_ids and questions must have identical lengths")
    total = len(questions)
    search_results: list[dict] = []

    effective_bench_size = max(1, bench_size)
    effective_concurrency = max(1, max_concurrency)
    # 信号量控制并发；bench_size 只影响回调触发频率，与并发度正交
    semaphore = asyncio.Semaphore(effective_concurrency)

    # 预建 engine 池，避免每次查询都重复构建 PipelineEngine
    # （engine.__init__ 会构建 DocumentLoader/EventExtractor/SAGSearcher 等重型对象）
    # 注意：engine.result 是可变状态，不能跨并发查询共享同一实例，
    # 因此池大小设为 effective_concurrency，每个并发槽独占一个 engine。
    engine_pool: asyncio.Queue[Any] = asyncio.Queue()
    engines: list[Any] = []
    if strategy != "multi_es":
        for _ in range(effective_concurrency):
            engine = PipelineEngine(
                task_config=TaskConfig(
                    task_name="benchmark",
                    source_config_id=source_config_id,
                ),
                auto_setup_logging=False,
            )
            engines.append(engine)
            engine_pool.put_nowait(engine)

    async def search_one(idx: int, question: str) -> dict:
        async with semaphore:
            try:
                if multi_es_searcher is not None:
                    raw = await multi_es_searcher.search_for_sections(
                        query=question,
                        source_config_ids=[source_config_id],
                        config=strategy_config,
                    )
                    raw_sections = raw.get("sections", [])
                    sections = [normalize_section(s) for s in raw_sections]
                    sections = sections[:top_k]
                    return {
                        "question_index": idx + 1,
                        "question": question,
                        "dataset_sample_id": sample_ids[idx],
                        "sections": sections,
                        "timing_steps": raw.get("_timings", {}),
                        "clues": [],
                        "nodes": {},
                        "route_index": {},
                        "evidence_coverage": {},
                        "event_stats": {},
                    }

                # 从池中取出 engine，用完放回（池大小 == 并发数，不会阻塞）
                engine = await engine_pool.get()
                try:
                    await engine.search_async(
                        SearchBaseConfig(
                            query=question,
                            return_type=ReturnType.PARAGRAPH,
                            rerank=RerankConfig(strategy=search_strategy),
                            strategy_config=strategy_config,
                            gold_evidences=(
                                gold_docs_for_recall[idx]
                                if idx < len(gold_docs_for_recall)
                                else None
                            ),
                        )
                    )
                    engine_result = engine.get_result()
                    raw_sections = (
                        engine_result.search_result.data_full
                        if engine_result and engine_result.search_result
                        else []
                    )
                    # 支持分阶段计时的策略会在此透传；其它策略为空 dict。
                    timing_steps = (
                        engine_result.search_result.stats.get("timing_steps", {})
                        if engine_result and engine_result.search_result
                        else {}
                    )
                    # ── 追链数据（SAG2 策略有值，其它策略为空）──
                    stats = (
                        engine_result.search_result.stats
                        if engine_result and engine_result.search_result
                        else {}
                    )
                    clues = stats.get("clues", [])
                    nodes = stats.get("nodes", {})
                    route_index = stats.get("route_index", {})
                    evidence_coverage = stats.get("evidence_coverage", {})
                    rank_event_ids = stats.get("rank_event_ids", [])
                    display_event_ids = stats.get("display_event_ids", [])
                    rank_scores = stats.get("rank_scores", {})
                    score_top_event_ids = stats.get("score_top_event_ids", [])
                    expand_hops = stats.get("expand_hops", [])
                    recall_clue_stats = stats.get("recall_clue_stats", {})
                    chunk_headings = stats.get("chunk_headings", {})
                    search_stats = stats.get("search_stats", {})
                    event_stats = (
                        search_stats.get("event_stats", {})
                        if isinstance(search_stats, dict)
                        else {}
                    )
                    gold_docs = gold_docs_for_recall[idx] if idx < len(gold_docs_for_recall) else []
                finally:
                    engine_pool.put_nowait(engine)

                sections = [normalize_section(s) for s in raw_sections]
                sections = sections[:top_k]
                return {
                    "question_index": idx + 1,
                    "question": question,
                    "dataset_sample_id": sample_ids[idx],
                    "sections": sections,
                    "timing_steps": timing_steps,
                    "clues": clues,
                    "nodes": nodes,
                    "route_index": route_index,
                    "evidence_coverage": evidence_coverage,
                    "rank_event_ids": rank_event_ids,
                    "display_event_ids": display_event_ids,
                    "rank_scores": rank_scores,
                    "score_top_event_ids": score_top_event_ids,
                    "expand_hops": expand_hops,
                    "recall_clue_stats": recall_clue_stats,
                    "chunk_headings": chunk_headings,
                    "event_stats": event_stats,
                    "gold_docs": gold_docs,
                }
            except Exception as e:
                logger.warning(f"问题 {idx + 1} 搜索失败: {e}")
                return {
                    "question_index": idx + 1,
                    "question": question,
                    "dataset_sample_id": sample_ids[idx],
                    "sections": [],
                    "timing_steps": {},
                    "clues": [],
                    "nodes": {},
                    "route_index": {},
                    "evidence_coverage": {},
                    "event_stats": {},
                }

    # 所有查询同时创建，靠 semaphore 限流到 effective_concurrency 真正并发；
    # bench_size 只控制进度回调的触发频率，不再限制并发度。
    # 结果按 idx 存入预分配数组，保证最终顺序与 questions / gold_docs 对齐。
    ordered_results: list[dict | None] = [None] * total
    completed_count = 0
    completed_lock = asyncio.Lock()
    # 跨 _emit_progress 调用的「上次累积 sums/counts 快照」，用于算「当批 = 本次累积 − 上次累积」。
    # 并发乱序完成，不能按 idx 切片，故用增量法。stats_lock 保证「读旧→算增量→写回新」原子。
    prev_stats: dict[str, Any] = {}
    stats_lock = asyncio.Lock()
    # Serialize progress checkpoints in the order they are created.  Without
    # this outer lock, a later (larger) cumulative snapshot could reach the raw
    # delta lock before an earlier checkpoint and make the batch subtraction
    # negative even though individual result completion is correctly indexed.
    progress_lock = asyncio.Lock()

    async def run_one(idx: int, question: str) -> None:
        nonlocal completed_count
        result = await search_one(idx, question)
        # 在锁内：写入结果、累加计数、判断是否到回调点、拍下自洽快照
        snapshot: list[dict | None] | None = None
        snapshot_idx = 0
        async with completed_lock:
            ordered_results[idx] = result
            completed_count += 1
            done = completed_count
            if done % effective_bench_size == 0 or done == total:
                # 浅拷贝当前已完成视图：此刻 done 与非 None 项数量一致，
                # 后续并发完成不会改写这份快照，避免 current_idx 与统计对不上
                snapshot = list(ordered_results)
                snapshot_idx = done

        # 锁外执行日志/指标计算（基于不可变快照，无竞态）
        if snapshot is not None:
            async with progress_lock:
                await _emit_progress(
                    current_idx=snapshot_idx,
                    total=total,
                    ordered_results=snapshot,
                    gold_docs_for_recall=gold_docs_for_recall,
                    bench_size=effective_bench_size,
                    mlflow_tracker=mlflow_tracker,
                    bench_logger=bench_logger,
                    prev_stats=prev_stats,
                    stats_lock=stats_lock,
                )

    try:
        await asyncio.gather(*(run_one(i, questions[i]) for i in range(total)))

        # 收尾：把所有结果按顺序填入 search_results
        for r in ordered_results:
            if r is not None:
                search_results.append(r)

        return search_results
    finally:
        # engine owns SAGSearcher -> SAG2Searcher -> SAG2Runtime.  Close that
        # chain before the process-scoped AI/storage singletons are released.
        close_results = await asyncio.gather(
            *(engine.aclose() for engine in engines),
            return_exceptions=True,
        )
        for result in close_results:
            if isinstance(result, BaseException):
                logger.warning("关闭搜索引擎运行时失败: %s", result)


def _collect_recall_stats(
    ordered_results: list[dict | None],
    gold_docs_for_recall: list[list[str]],
) -> dict[str, Any]:
    """收集所有已完成查询的 retrieved/gold docs 和 full/partial/zero 统计。

    Returns:
        {"retrieved_docs_list": [...], "matched_gold_docs": [...],
         "full": int, "partial": int, "zero": int, "completed": int, "denom": int}
    """
    retrieved_docs_list: list[list[str]] = []
    matched_gold_docs: list[list[str]] = []
    full_count = partial_count = zero_count = 0
    for idx, result in enumerate(ordered_results):
        if result is None:
            continue
        sections = result.get("sections", [])
        gold_docs = gold_docs_for_recall[idx]
        retrieved_docs_list.append(sections)
        matched_gold_docs.append(gold_docs)
        matched = [doc for doc in sections if doc in gold_docs]
        if len(matched) == 0:
            zero_count += 1
        elif len(matched) < len(gold_docs):
            partial_count += 1
        else:
            full_count += 1
    completed = full_count + partial_count + zero_count
    return {
        "retrieved_docs_list": retrieved_docs_list,
        "matched_gold_docs": matched_gold_docs,
        "full": full_count,
        "partial": partial_count,
        "zero": zero_count,
        "completed": completed,
        "denom": completed if completed > 0 else 1,
    }


def _compute_batch_view(
    cum: dict[str, Any],
    prev_stats: dict[str, Any],
) -> dict[str, Any] | None:
    """根据累积 raw 快照和上次 raw 快照计算当批增量视图。

    调用方必须已经持有 stats_lock。prev_stats 会在本函数内原地更新为本次累积值。
    schema v2 与 legacy 都复用 step_timing 中的纯差分函数；并发完成顺序不参与切片。
    """
    batch_view = batch_view_from_cumulative(cum, prev_stats)
    snapshot = raw_diagnostic_snapshot(cum)
    prev_stats.clear()
    prev_stats.update(snapshot)
    return batch_view


def _log_and_report_step_timings(
    bench_logger,
    cum: dict[str, Any],
    batch_view: dict[str, Any] | None,
    mlflow_tracker: Any | None,
    batch_index: int,
) -> None:
    """在核心指标之后按固定顺序展示并上报新增诊断。"""
    if cum.get("schema_version") == 2:
        # Schema v2 timing/token diagnostics remain persisted in MLflow and
        # the result payload, but are intentionally silent in the console.
        if mlflow_tracker is not None:
            supplementary = build_supplementary_metrics(cum, batch_view)
            if supplementary:
                mlflow_tracker.log_supplementary_metrics(supplementary, batch_index)
        return

    # Timing/token metrics are persisted and reported to MLflow, but are not
    # printed per stage in the console.
    if mlflow_tracker is None:
        return
    if _has_step_timing(cum):
        mlflow_tracker.log_timing_metrics(
            _build_timing_mlflow_metrics(cum, suffix="_cum"), batch_index
        )
        mlflow_tracker.log_token_metrics(
            _build_token_mlflow_metrics(cum, suffix="_cum"), batch_index
        )
    if batch_view is not None and _has_step_timing(batch_view):
        mlflow_tracker.log_token_metrics(
            _build_token_mlflow_metrics(batch_view, suffix="_batch"), batch_index
        )


async def _emit_progress(
    current_idx: int,
    total: int,
    ordered_results: list[dict | None],
    gold_docs_for_recall: list[list[str]],
    bench_size: int,
    mlflow_tracker: Any | None,
    bench_logger,
    prev_stats: dict[str, Any] | None = None,
    stats_lock: Any | None = None,
) -> None:
    """
    进度回调：统计截至当前所有已完成查询（ordered_results 中非 None 项）的召回。

    与旧 batch 实现的语义等价——统计"已完成的全部查询"的召回情况。
    Recall@K 同样基于已完成查询的全量 retrieved/gold docs。
    并发完成顺序不影响统计（每项用自身 idx 对齐对应的 gold_docs）。
    """
    batch_index = (current_idx + bench_size - 1) // bench_size
    total_batches = (total + bench_size - 1) // bench_size

    bench_logger.info(f"\n{'=' * 80}")
    bench_logger.info(
        f"📝 Bench 进度: 批次 {batch_index}/{total_batches} ({current_idx}/{total} 问题)"
    )
    bench_logger.info(f"{'=' * 80}")

    # ── 召回统计 ────────────────────────────────────────────────
    rs = _collect_recall_stats(ordered_results, gold_docs_for_recall)
    bench_logger.info(f"\n📊 累积召回情况统计 ({rs['completed']} 个问题):")
    bench_logger.info("=" * 50)
    bench_logger.info(f"✅ 全部召回: {rs['full']} 个 ({rs['full'] / rs['denom'] * 100:.1f}%)")
    bench_logger.info(f"⚠️  部分召回: {rs['partial']} 个 ({rs['partial'] / rs['denom'] * 100:.1f}%)")
    bench_logger.info(f"❌ 零召回: {rs['zero']} 个 ({rs['zero'] / rs['denom'] * 100:.1f}%)")
    bench_logger.info("=" * 50)

    recall_metric = RetrievalRecall()
    pooled_recall, _ = recall_metric.calculate_metric_scores(
        gold_docs=rs["matched_gold_docs"],
        retrieved_docs=rs["retrieved_docs_list"],
        k_list=[1, 2, 5, 10],
    )
    pooled_precision_f1 = calculate_precision_f1_at_k(
        gold_docs=rs["matched_gold_docs"],
        retrieved_docs=rs["retrieved_docs_list"],
        k_list=[2, 5, 10],
    )
    bench_logger.info("\n✅ 累积Recall@K:")
    for metric, score in pooled_recall.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score * 100:.2f}%)")
    bench_logger.info("\n✅ 累积Precision/F1@K:")
    for metric, score in pooled_precision_f1.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score * 100:.2f}%)")

    # ── 构造新增诊断（先不展示/上报；核心指标必须先写 MLflow）──
    cum = compute_step_timings(ordered_results)
    batch_view: dict[str, Any] | None = None
    if prev_stats is not None and stats_lock is not None:
        async with stats_lock:
            batch_view = _compute_batch_view(cum, prev_stats)
    if mlflow_tracker:
        pooled_results = {**pooled_recall, **pooled_precision_f1}
        mlflow_tracker.log_evaluation_metrics(
            rs["full"], rs["partial"], rs["zero"], current_idx, step=batch_index
        )
        mlflow_tracker.log_recall_metrics(pooled_results, step=batch_index)

    # ── 新增时间/事件/Token 诊断最后展示和上报 ───────────────
    _log_and_report_step_timings(bench_logger, cum, batch_view, mlflow_tracker, batch_index)


def print_final_summary(
    dataset_name: str,
    strategy: str,
    questions: list[str],
    search_results: list[dict],
    gold_docs_for_recall: list[list[str]],
    k_values: list[int],
    search_time: float,
    bench_logger,
) -> dict[str, Any]:
    """
    打印最终汇总结果，格式与 benchmark.py 完全一致。
    返回包含所有指标的字典（供保存 JSON）。
    """
    total = len(search_results)

    # ── 召回统计 ──────────────────────────────────────────────
    full_count = partial_count = zero_count = 0
    retrieved_docs_list = []

    for i, result in enumerate(search_results):
        sections = result.get("sections", [])
        retrieved_docs_list.append(sections)
        if i < len(gold_docs_for_recall):
            gold_docs = gold_docs_for_recall[i]
            matched = [doc for doc in sections if doc in gold_docs]
            if len(matched) == 0:
                zero_count += 1
            elif len(matched) < len(gold_docs):
                partial_count += 1
            else:
                full_count += 1

    # ── Recall@K ──────────────────────────────────────────────
    recall_metric = RetrievalRecall()
    pooled_recall, _ = recall_metric.calculate_metric_scores(
        gold_docs=gold_docs_for_recall[:total], retrieved_docs=retrieved_docs_list, k_list=k_values
    )

    # ── Precision@K / F1@K ────────────────────────────────────
    precision_f1_k_list = [k for k in k_values if k >= 2]
    if not precision_f1_k_list:
        precision_f1_k_list = k_values
    pooled_precision_f1 = calculate_precision_f1_at_k(
        gold_docs=gold_docs_for_recall[:total],
        retrieved_docs=retrieved_docs_list,
        k_list=precision_f1_k_list,
    )

    # ── 打印（与 benchmark.py _handle_bench_callback 格式完全一致）──
    bench_logger.info(f"\n{'=' * 80}")
    bench_logger.info(f"📝 最终评估结果: 数据集={dataset_name}, 策略={strategy}, 共 {total} 个问题")
    bench_logger.info(f"{'=' * 80}")

    bench_logger.info(f"\n📊 最终召回情况统计 ({total} 个问题):")
    bench_logger.info("=" * 50)
    bench_logger.info(f"✅ 全部召回: {full_count} 个 ({full_count / total * 100:.1f}%)")
    bench_logger.info(f"⚠️  部分召回: {partial_count} 个 ({partial_count / total * 100:.1f}%)")
    bench_logger.info(f"❌ 零召回: {zero_count} 个 ({zero_count / total * 100:.1f}%)")
    bench_logger.info("=" * 50)

    bench_logger.info("\n✅ 最终Recall@K:")
    for metric, score in pooled_recall.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score * 100:.2f}%)")

    bench_logger.info("\n✅ 最终Precision/F1@K:")
    for metric, score in pooled_precision_f1.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score * 100:.2f}%)")

    # 保留各 Step 耗时计算，供结果文件和 MLflow 使用；不在控制台逐阶段打印。
    timings = compute_step_timings(search_results)

    bench_logger.info("\n阶段耗时统计:")
    bench_logger.info("=" * 50)
    bench_logger.info(f"  SEARCH阶段: {search_time:.1f} 秒")
    bench_logger.info(f"  总计: {search_time:.1f} 秒")
    bench_logger.info("=" * 50)

    total_successful = sum(1 for r in search_results if r.get("sections"))
    bench_logger.info(f"\n检索统计: {total_successful}/{total} 个问题检索成功")
    bench_logger.info(f"{'=' * 80}\n")

    return {
        "recall": pooled_recall,
        "precision_f1": pooled_precision_f1,
        "timings": timings,
        "statistics": {
            "total_questions": total,
            "full_recall_count": full_count,
            "partial_recall_count": partial_count,
            "zero_recall_count": zero_count,
            "successful_searches": total_successful,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="搜索 + Benchmark 评估（展示/计分逻辑与 benchmark.py 完全一致）"
    )

    # 必填
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="数据集名称 (musique, hotpotqa, test_hotpotqa 等)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=[
            "atomic",
            "multi_es",
            "sag2",
            "vector",
            "bm25",
        ],
        help="搜索策略",
    )

    # 检索参数
    parser.add_argument("--top-k", type=int, default=10, help="返回前K个结果（默认：10）")
    parser.add_argument(
        "--sag2-rerank-top-k",
        type=int,
        default=None,
        help="SAG2 rerank 模型真正保留的最高 K 个结果（默认读取配置：10）",
    )
    parser.add_argument(
        "--sag2-max-results",
        type=int,
        default=None,
        help="SAG2 最终事项数；rerank 不足时按 embedding 相似度补齐（默认读取配置：10）",
    )
    parser.add_argument(
        "--k-values", type=str, default="1,2,5,10", help="评估的K值列表，逗号分隔（默认：1,2,5,10）"
    )
    parser.add_argument("--max-concurrency", type=int, default=10, help="搜索并发数（默认：10）")
    parser.add_argument(
        "--limit",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="限制处理范围：--limit N 只处理前N条；--limit S E 处理第S到第E条（0-based，含两端）",
    )
    parser.add_argument(
        "--bench-size", type=int, default=5, help="每 N 个问题打印一次累积统计（默认：5）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="SAG2 策略专属：在日志中打印 LLM 精选步骤（_llm_rank_events）和 query 重写步骤"
        "（_rewrite_query_and_extract_entities）的完整 prompt 输入和输出内容。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "precise"],
        default="fast",
        help="multi_es mode: fast=BM25实体召回+小规模扩展, precise=BM25实体召回+LLM filter",
    )

    parser.add_argument(
        "--sag2-scope",
        action="store_true",
        help="启用 SAG2 事件候选池搜索变体（默认关闭；与 vector、atomic 等策略并列比较）。",
    )
    parser.add_argument(
        "--sag2-event-top-k",
        type=int,
        default=1000,
        help="SAG2 候选池大小 k_pool（按 query 相似度选取的事件数，默认：1000）。",
    )
    parser.add_argument(
        "--sag2-bootstrap-entity-limit",
        type=int,
        default=0,
        help="Maximum event-entity relations loaded during bootstrap (0 = unlimited).",
    )
    parser.add_argument(
        "--sag2-no-event-content",
        action="store_true",
        help="Do not keep event content in the in-memory universe.",
    )
    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：output/<数据集>/<策略>/<时间戳>/）",
    )

    # 数据源
    parser.add_argument(
        "--source-config-id",
        type=str,
        default=None,
        help="直接指定 source_config_id，跳过自动文件查找",
    )

    parser.add_argument(
        "--allow-embedding-mismatch",
        action="store_true",
        help="允许当前 embedding 维度与建库时不同（不推荐，结果不可比）",
    )

    # MLflow
    parser.add_argument("--use-mlflow", action="store_true", help="启用 MLflow 跟踪")
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=None,
        help="MLflow Tracking Server 地址（默认使用 .env 中的 MLFLOW_URL）",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="sag-benchmark",
        help="MLflow 实验名称（默认：sag-benchmark）",
    )
    # MLflow Prompt Registry（提示词管理，独立于上面的 tracking）
    parser.add_argument(
        "--use-mlflow-prompts",
        action="store_true",
        help="从 MLflow Prompt Registry 加载检索提示词（启动时一次性加载并打印来源；"
        "未注册的条目回退到代码常量，服务不可达则报错）。默认关闭=用代码常量。"
        "tracking URI 复用 --mlflow-url 的地址。",
    )
    parser.add_argument(
        "--mlflow-prompt-alias",
        type=str,
        default="latest",
        help="MLflow Prompt Registry 别名（--use-mlflow-prompts 时生效，默认 latest）。"
        "加载时使用 prompts:/name@{alias}。",
    )
    args = parser.parse_args()

    # ── SAG2 verbose 模式 ──────────────────────────────────────
    if args.verbose:
        from pipeline.modules.search.sag2 import SAG2Searcher

        SAG2Searcher.verbose = True
        logger.info("🔊 SAG2 verbose 模式已启用：将打印 LLM 精选步骤的完整 prompt 输入和输出")
    if args.strategy == "sag2":
        try:
            build_sag2_config(
                max_sections=args.top_k,
                rerank_top_k=args.sag2_rerank_top_k,
                max_results=args.sag2_max_results,
                scope_enabled=args.sag2_scope,
                event_top_k=args.sag2_event_top_k,
                bootstrap_entity_limit=args.sag2_bootstrap_entity_limit,
                include_event_content=not args.sag2_no_event_content,
            )
        except ValueError as exc:
            parser.error(f"SAG2 rerank 参数无效: {exc}")

    # 读取 .env 中的 LLM_MODEL 配置
    settings = get_settings()
    llm_model = settings.llm_model
    logger.info(f"📌 当前 LLM 模型: {llm_model}")

    # 解析 K 值
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # 确定输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "output" / args.dataset_name / args.strategy / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 将所有日志同时写入 output_dir/run.log ─────────────────
    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    # DEBUG 级：让放开到 DEBUG 的 logger（如 pipeline.ai.openai 的完整请求/响应）
    # 能落盘；其余 logger 因自身级别限制不会产生 DEBUG record，不会被这里放大。
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # 挂到根 logger，确保所有子 logger 的日志都能落盘
    logging.getLogger().addHandler(file_handler)
    logger.info(f"📄 日志文件: {log_file}")

    # bench_logger：使用 logger（与 benchmark.py 一致用 logging 而非 print）
    bench_logger = logging.getLogger("scripts.run_search_benchmark")

    # ── 打印启动信息 ──────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("🔍 检索信息概览")
    logger.info("=" * 80)
    logger.info(f"📊 数据集名称: {args.dataset_name}")
    logger.info(f"🔧 检索策略: {args.strategy}")
    if args.strategy == "multi_es":
        logger.info(f"  multi_es mode: {args.mode}")
    if args.strategy == "sag2":
        logger.info(
            "  SAG2 scope: enabled=%s, event_top_k=%d, bootstrap_entity_limit=%d, include_event_content=%s",
            args.sag2_scope,
            args.sag2_event_top_k,
            args.sag2_bootstrap_entity_limit,
            not args.sag2_no_event_content,
        )
    logger.info(f"  Top-K: {args.top_k}")
    if args.strategy == "sag2":
        effective_sag2_config = build_sag2_config(
            max_sections=args.top_k,
            rerank_top_k=args.sag2_rerank_top_k,
            max_results=args.sag2_max_results,
            scope_enabled=args.sag2_scope,
            event_top_k=args.sag2_event_top_k,
            bootstrap_entity_limit=args.sag2_bootstrap_entity_limit,
            include_event_content=not args.sag2_no_event_content,
        )
        logger.info(
            "  SAG2 rerank: top_k=%d, max_results=%d",
            effective_sag2_config.sag2_rerank.rerank_top_k,
            effective_sag2_config.sag2_rerank.max_results,
        )
    logger.info(f"  K值列表: {k_values}")
    logger.info(f"  并发数: {args.max_concurrency}")
    logger.info(f"  Bench size: {args.bench_size}")

    # ── 解析 --limit 参数 ──────────────────────────────────────
    limit_start: int | None = None
    limit_end: int | None = None  # 切片用的 stop（exclusive）
    if args.limit:
        if len(args.limit) == 1:
            limit_end = args.limit[0]
            logger.info(f"  限制条数: 前 {limit_end} 条（索引 0~{limit_end - 1}）")
        elif len(args.limit) == 2:
            limit_start, limit_end_incl = args.limit[0], args.limit[1]
            if limit_start > limit_end_incl:
                logger.error(
                    f"--limit 起始索引 {limit_start} > 结束索引 {limit_end_incl}，请检查参数"
                )
                sys.exit(1)
            limit_end = limit_end_incl + 1  # 转为 exclusive
            logger.info(
                f"  限制范围: 第 {limit_start} 条 ~ 第 {limit_end_incl} 条（共 {limit_end - limit_start} 条）"
            )
        else:
            logger.error("--limit 最多接受两个整数")
            sys.exit(1)

    logger.info(f"  输出目录: {output_dir}")
    logger.info("\n" + "=" * 80)

    # ── 加载数据集 ────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_loader = DatasetLoader(args.dataset_name)
    question_records = dataset_loader.get_question_records()
    if args.limit:
        question_records = question_records[limit_start:limit_end]
    questions = [record["question"] for record in question_records]
    sample_ids = [record["id"] for record in question_records]

    gold_docs_for_recall = dataset_loader.get_gold_docs_for_recall(limit=None)
    if gold_docs_for_recall and args.limit:
        gold_docs_for_recall = gold_docs_for_recall[limit_start:limit_end]

    logger.info(f"Successfully loaded dataset with {len(question_records)} questions")
    logger.info(f"❓ 问题总数: {len(questions)}")
    logger.info(f"📋 数据范围: [0:{len(questions)}]，共 {len(questions)} 个问题:")
    logger.info("\n" + "=" * 80)

    # ── 获取 source_config_id ─────────────────────────────────
    if args.source_config_id:
        source_config_id = args.source_config_id
        source_timestamp = "manual"
        logger.info(f"📦 使用数据源（手动指定）: {source_config_id}")
    else:
        try:
            source_info = load_latest_source_info(args.dataset_name, llm_model)
            source_config_id = source_info["source_config_id"]
            source_timestamp = source_info.get("timestamp", "unknown")

            # 增强日志输出，明确显示使用的数据源信息
            logger.info(f"📦 使用数据源: {source_config_id}")
            logger.info(f"   模型: {source_info.get('model_name', 'unknown')}")
            logger.info(f"   时间戳: {source_timestamp}")
            logger.info(f"   文件路径: {source_info.get('file_path', 'unknown')}")
        except Exception as e:
            logger.error(f"无法获取 source_config_id: {e}")
            sys.exit(1)

    # ── 设定 embedding 维度（双层路由：索引用维度，过滤用 source_config_id）──
    from pipeline.modules.search.benchmark_utils import activate_embedding_dim_for_source

    source_info_for_dim = source_info if not args.source_config_id else None
    await activate_embedding_dim_for_source(
        source_info_for_dim, allow_mismatch=args.allow_embedding_mismatch
    )

    # ── 解析 MLflow URL（--use-mlflow 和 --use-mlflow-prompts 共用）──
    mlflow_url = args.mlflow_url or settings.mlflow_url

    # ── 初始化 MLflow ─────────────────────────────────────────
    mlflow_tracker = None
    if args.use_mlflow:
        try:
            logger.info(f"📊 MLflow Tracking Server: {mlflow_url}")

            mlflow_config = MLflowConfig(
                uri=mlflow_url,
                experiment=f"{get_local_ip()}_{args.mlflow_experiment}",
                dataset_name=args.dataset_name,
                bench_size=args.bench_size,
                enable_qa=False,
            )
            mlflow_tracker = MLflowTracker(mlflow_config, questions)
            mlflow_tracker.start()

            # ── 记录运行命令 + 脚本参数 ──────────────────────────

            import mlflow

            cmd_parts = [
                "uv run python scripts/run_search_benchmark.py",
                f"--dataset-name {args.dataset_name}",
                f"--strategy {args.strategy}",
                f"--mode {args.mode}",
                f"--top-k {args.top_k}",
                f'--k-values "{args.k_values}"',
                f"--max-concurrency {args.max_concurrency}",
                f"--bench-size {args.bench_size}",
            ]
            if args.limit:
                cmd_parts.append(f"--limit {' '.join(str(x) for x in args.limit)}")
            if args.output_dir:
                cmd_parts.append(f"--output-dir {args.output_dir}")
            if args.source_config_id:
                cmd_parts.append(f"--source-config-id {args.source_config_id}")
            if args.strategy == "sag2":
                if args.sag2_scope:
                    cmd_parts.append("--sag2-scope")
                cmd_parts.append(f"--sag2-event-top-k {args.sag2_event_top_k}")
                cmd_parts.append(
                    f"--sag2-bootstrap-entity-limit {args.sag2_bootstrap_entity_limit}"
                )
                if args.sag2_no_event_content:
                    cmd_parts.append("--sag2-no-event-content")
            if args.sag2_rerank_top_k is not None:
                cmd_parts.append(f"--sag2-rerank-top-k {args.sag2_rerank_top_k}")
            if args.sag2_max_results is not None:
                cmd_parts.append(f"--sag2-max-results {args.sag2_max_results}")
            if args.use_mlflow:
                cmd_parts += [
                    "--use-mlflow",
                    f"--mlflow-url {args.mlflow_url}",
                    f"--mlflow-experiment {args.mlflow_experiment}",
                ]
            mlflow.log_param("run_command", " \\\n    ".join(cmd_parts))

            # ── 记录策略实际生效的默认配置 ────────────────────────
            from pipeline.modules.search.config import (
                AtomicConfig,
                BM25Config,
                MultiConfig,
                VectorConfig,
            )

            if args.strategy == "multi_es":
                cfg = MultiConfig(
                    strategy="multi_es",
                    mode=args.mode,
                    max_sections=args.top_k,
                )
                strategy_defaults = {
                    "strategy": "multi_es",
                    "mode": cfg.mode,
                    "entity_top_k": cfg.entity_top_k,
                    "multi_top_k": cfg.multi_top_k,
                    "precise_entity_event_top_k": 40,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_hops": cfg.max_hops,
                    "max_events": cfg.max_events,
                    "max_expand_events_per_hop": cfg.max_expand_events_per_hop,
                    "max_sections": cfg.max_sections,
                    "fast_entity_k": cfg.fast_entity_k,
                    "fast_entity_event_candidate_k": cfg.fast_entity_event_candidate_k,
                    "fast_entity_event_k": cfg.fast_entity_event_k,
                    "fast_query_event_k": cfg.fast_query_event_k,
                    "fast_answer_k": cfg.fast_answer_k,
                    "fast_expand_answer_k": cfg.fast_expand_answer_k,
                    "fast_vector_weight": cfg.fast_vector_weight,
                    "fast_entity_weight": cfg.fast_entity_weight,
                    "fast_channel_weight": cfg.fast_channel_weight,
                }
            elif args.strategy == "sag2":
                cfg = build_sag2_config(
                    max_sections=args.top_k,
                    rerank_top_k=args.sag2_rerank_top_k,
                    max_results=args.sag2_max_results,
                    scope_enabled=args.sag2_scope,
                    event_top_k=args.sag2_event_top_k,
                    bootstrap_entity_limit=args.sag2_bootstrap_entity_limit,
                    include_event_content=not args.sag2_no_event_content,
                )
                strategy_defaults = {
                    "strategy": "sag2",
                    "sag2_event_top_k": cfg.sag2_scope.event_top_k,
                    "sag2_scope_enabled": cfg.sag2_scope.enabled,
                    "sag2_bootstrap_entity_limit": cfg.sag2_scope.bootstrap_entity_limit,
                    "sag2_include_event_content": cfg.sag2_scope.include_event_content,
                    "sag2_max_entities": cfg.sag2_recall.max_entities,
                    "sag2_max_hops": cfg.sag2_expand.max_hops,
                    "sag2_rerank_strategy": cfg.sag2_rerank.strategy,
                    "sag2_rerank_top_k": cfg.sag2_rerank.rerank_top_k,
                    "sag2_max_results": cfg.sag2_rerank.max_results,
                    "max_sections": cfg.max_sections,
                }
            elif args.strategy == "atomic":
                cfg = AtomicConfig(max_sections=args.top_k)
                strategy_defaults = {
                    "strategy": "atomic",
                    "entity_top_k": cfg.entity_top_k,
                    "atomic_top_k": cfg.atomic_top_k,
                    "key_similarity_threshold": cfg.key_similarity_threshold,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_hops": cfg.max_hops,
                    "max_events": cfg.max_events,
                    "rerank_top_k": cfg.rerank_top_k,
                    "max_sections": cfg.max_sections,
                }
            elif args.strategy == "bm25":
                cfg = BM25Config(top_k=args.top_k)
                strategy_defaults = {
                    "strategy": "bm25",
                    "top_k": cfg.top_k,
                    "similarity_threshold": cfg.similarity_threshold,
                }
            else:  # vector
                cfg = VectorConfig(top_k=args.top_k)
                strategy_defaults = {
                    "strategy": "vector",
                    "top_k": cfg.top_k,
                    "title_weight": cfg.title_weight,
                    "content_weight": cfg.content_weight,
                    "similarity_threshold": cfg.similarity_threshold,
                }
            mlflow.log_params({f"cfg_{k}": v for k, v in strategy_defaults.items()})

            # ── 记录模型和数据源信息 ──────────────────────────────
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("source_config_id", source_config_id)
            mlflow.log_param("source_timestamp", source_timestamp)

            logger.info("✅ MLflow 追踪器初始化完成")
        except Exception as e:
            logger.warning(f"MLflow 初始化失败: {e}")
            mlflow_tracker = None

    token_tracker = LLMTokenTracker()
    logger.info("✅ LLM token 追踪已启用（当前运行上下文）")

    # ── 执行搜索 ──────────────────────────────────────────────
    logger.info(f"\n🚀 启动检索 (策略: {args.strategy})...")
    logger.info("=" * 80)

    search_start = time.perf_counter()
    with llm_tracking_scope(token_tracker), llm_tracking_stage("SEARCH"):
        search_results = await run_batch_search(
            sample_ids=sample_ids,
            questions=questions,
            source_config_id=source_config_id,
            strategy=args.strategy,
            mode=args.mode,
            top_k=args.top_k,
            max_concurrency=args.max_concurrency,
            bench_size=args.bench_size,
            gold_docs_for_recall=gold_docs_for_recall,
            mlflow_tracker=mlflow_tracker,
            bench_logger=bench_logger,
            use_mlflow_prompts=args.use_mlflow_prompts,
            mlflow_prompt_alias=args.mlflow_prompt_alias,
            mlflow_url=mlflow_url,
            sag2_scope_enabled=args.sag2_scope,
            sag2_event_top_k=args.sag2_event_top_k,
            sag2_bootstrap_entity_limit=args.sag2_bootstrap_entity_limit,
            sag2_include_event_content=not args.sag2_no_event_content,
            sag2_rerank_top_k=args.sag2_rerank_top_k,
            sag2_max_results=args.sag2_max_results,
        )
    search_time = time.perf_counter() - search_start

    # ── 保存搜索原始结果 ──────────────────────────────────────
    search_output = output_dir / "search_results.json"
    serializable = [
        {
            "question_index": r["question_index"],
            "question": r["question"],
            "retrieved_docs": r["sections"],
            "dataset_sample_id": r["dataset_sample_id"],
        }
        for r in search_results
    ]
    with open(search_output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"\n💾 搜索结果已保存: {search_output}")

    # ── 打印最终汇总（与 benchmark.py 完全一致）──────────────
    metrics = print_final_summary(
        dataset_name=args.dataset_name,
        strategy=args.strategy,
        questions=questions,
        search_results=search_results,
        gold_docs_for_recall=gold_docs_for_recall,
        k_values=k_values,
        search_time=search_time,
        bench_logger=bench_logger,
    )

    # ── 保存 benchmark 结果 ───────────────────────────────────
    token_summary = token_tracker.get_summary()
    benchmark_output = output_dir / "benchmark_results.json"
    with open(benchmark_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {**metrics["recall"], **metrics["precision_f1"]},
                "statistics": metrics["statistics"],
                "timings": metrics.get("timings", {}),
                "search_diagnostics": public_search_diagnostics(metrics.get("timings", {})),
                "llm_token_usage": token_summary,
                "metadata": {
                    "dataset_name": args.dataset_name,
                    "strategy": args.strategy,
                    "mode": args.mode if args.strategy == "multi_es" else None,
                    "top_k": args.top_k,
                    "k_values": k_values,
                    "total_questions": len(search_results),
                    "search_time_seconds": round(search_time, 2),
                    "timestamp": timestamp,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"💾 评估结果已保存: {benchmark_output}")

    # ── MLflow 最终指标上报 ────────────────────────────────────
    if mlflow_tracker:
        try:
            import mlflow

            all_metrics = {**metrics["recall"], **metrics["precision_f1"]}

            # 计算最终的 batch_index，与中间记录保持一致
            final_batch_index = (len(search_results) + args.bench_size - 1) // args.bench_size

            stats = metrics["statistics"]
            mlflow_tracker.log_evaluation_metrics(
                stats["full_recall_count"],
                stats["partial_recall_count"],
                stats["zero_recall_count"],
                len(search_results),
                step=final_batch_index,
            )
            mlflow_tracker.log_recall_metrics(all_metrics, step=final_batch_index)
            mlflow.log_metric("search_time_seconds", search_time)
            mlflow.log_param("output_dir", str(output_dir))

            # 新增诊断统一最后上报；schema v2 复用中间批次的 ordered builder。
            final_timings = metrics.get("timings", {})
            if _has_step_timing(final_timings):
                if final_timings.get("schema_version") == 2:
                    mlflow_tracker.log_supplementary_metrics(
                        build_supplementary_metrics(final_timings),
                        step=final_batch_index,
                    )
                else:
                    mlflow_tracker.log_timing_metrics(
                        _build_timing_mlflow_metrics(final_timings, suffix="_cum"),
                        step=final_batch_index,
                    )
                    mlflow_tracker.log_token_metrics(
                        _build_token_mlflow_metrics(final_timings, suffix="_cum"),
                        step=final_batch_index,
                    )
            logger.info("✓ MLflow metrics/params 记录完成")
        except Exception as e:
            logger.warning(f"MLflow 记录失败: {e}")
        finally:
            mlflow_tracker.end()

    logger.info(f"\n✅ 完成！总耗时: {search_time:.1f} 秒")

    # 优雅关闭异步连接（ES aiohttp 连接池 + MySQL aiomysql 连接）。
    # 若不在事件循环存活时显式关闭，退出时 __del__ 会在 loop 已关闭的情况下
    # 触发 "Unclosed client session" / "Event loop is closed" 噪声报错（无害但刷屏）。
    try:
        from pipeline.core.ai.factory import close_all_clients
        from pipeline.storage import close_storage_facade

        await close_all_clients()
        await close_storage_facade()
        logger.info("✓ 已关闭 AI / ES / MySQL 异步连接")
    except Exception as e:
        logger.warning(f"关闭异步连接时出错（可忽略）: {e}")


if __name__ == "__main__":
    # 配置日志：根 console handler 设为 DEBUG，使放开到 DEBUG 的 logger
    # （如 pipeline.ai.openai 的完整 LLM 请求/响应）能在终端显示。
    # 各 logger 自身级别仍独立控制（见文件顶部），不会因此放大其他模块的日志。
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
