import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 复用 benchmark 共享逻辑，与run_search_benchmark.py 行为一致
from _search_common import (
    SUPPORTED_STRATEGIES,
    load_latest_source_info,
    build_strategy_config,
    create_multi_es_searcher,
    search_one_question,
)
from pipeline.core.config import get_settings
from pipeline.utils import get_logger

logger = get_logger("scripts.run_search_only")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("pipeline").setLevel(logging.WARNING)
    logging.getLogger("pipeline.search.multi_es").setLevel(logging.INFO)
    logging.getLogger("pipeline.ai.llm").setLevel(logging.INFO)
    for noisy_logger_name in (
        "elastic_transport",
        "elastic_transport.transport",
    ):
        logging.getLogger(noisy_logger_name).setLevel(logging.ERROR)
    for noisy_logger_name in (
        "elasticsearch",
        "httpx",
        "httpcore",
        "urllib3",
    ):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)


def get_dataset_path_for_name(dataset_name: str) -> str:
    """根据数据集名称获取默认数据集路径"""
    return str(project_root / "pipeline" / "evaluation" / "dataset" / f"{dataset_name}.json")


async def main():
    parser = argparse.ArgumentParser(description="单独的搜索脚本 - 只执行检索")

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=SUPPORTED_STRATEGIES,
        help="搜索策略：atomic / multi / multi_es / multi1 / hopllm / vector",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "precise"],
        help="multi_es 策略的检索模式（默认: fast）",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="数据集名称",
    )

    parser.add_argument(
        "--source-config-id",
        type=str,
        default=None,
        help="信息源ID，不指定则按 .env 的 LLM_MODEL 自动查找最新上传",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出根目录（默认: output）",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回前K个结果（默认: 10）",
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="搜索并发数（默认: 1）",
    )

    parser.add_argument(
        "--limit",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="限制处理范围：--limit N 只处理前N条；--limit S E 处理第S到第E条（0-based，含两端）",
    )

    args = parser.parse_args()

    # ── 解析 --limit 参数（与 run_search_benchmark.py 语义一致）────────
    limit_start: Optional[int] = None
    limit_end: Optional[int] = None  # 切片用的 stop（exclusive）
    if args.limit:
        if len(args.limit) == 1:
            limit_end = args.limit[0]
        elif len(args.limit) == 2:
            limit_start, limit_end_incl = args.limit[0], args.limit[1]
            if limit_start > limit_end_incl:
                logger.error(
                    f"--limit 起始索引 {limit_start} > 结束索引 {limit_end_incl}，请检查参数"
                )
                return 1
            limit_end = limit_end_incl + 1  # 转为 exclusive
        else:
            logger.error("--limit 最多接受两个整数")
            return 1

    # ── 确定 source_config_id ─────────────────────────────────────
    settings = get_settings()
    llm_model = settings.llm_model
    if args.source_config_id:
        source_config_id = args.source_config_id
        print(f"📦 使用数据源（手动指定）: {source_config_id}")
    else:
        try:
            source_info = load_latest_source_info(args.dataset_name, llm_model)
        except FileNotFoundError as e:
            print(f"✗ 无法获取 source_config_id: {e}")
            return 1
        source_config_id = source_info["source_config_id"]
        print(f"📦 使用数据源: {source_config_id}")
        print(f"   模型: {llm_model}")
        print(f"   时间戳: {source_info.get('timestamp', 'unknown')}")
        print(f"   文件路径: {source_info.get('file_path', 'unknown')}")

    # ── 确定数据集路径并加载 ───────────────────────────────────────
    dataset_path = get_dataset_path_for_name(args.dataset_name)
    if not Path(dataset_path).exists():
        print(f"✗ 数据集文件不存在: {dataset_path}")
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[limit_start:limit_end]

    print(f"\n{'=' * 70}")
    print("搜索配置:")
    print(f"  策略: {args.strategy}")
    if args.strategy == "multi_es":
        print(f"  multi_es mode: {args.mode}")
    print(f"  数据集: {args.dataset_name}")
    print(f"  信息源: {source_config_id}")
    print(f"  问题数: {len(dataset)}")
    print(f"  Top-K: {args.top_k}")
    print(f"  并发数: {args.max_concurrency}")
    print(f"{'=' * 70}\n")

    # 创建输出目录（按时间戳和数据集名称）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = Path(args.output_dir) / "search_results" / args.dataset_name / timestamp
    output_subdir.mkdir(parents=True, exist_ok=True)

    # ── 构造策略配置（与 benchmark 一致）──────────────────────────
    search_strategy, strategy_config = build_strategy_config(
        args.strategy, top_k=args.top_k, mode=args.mode
    )
    multi_es_searcher = None
    if args.strategy == "multi_es":
        multi_es_searcher = await create_multi_es_searcher(strategy_config)

    # ── 执行搜索 ──────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    total = len(dataset)

    async def search_one(idx: int, item: Dict) -> Dict:
        async with semaphore:
            question = item.get("question", "")
            print(f"[{idx + 1}/{total}] 搜索: {question[:60]}...")
            query_start = time.perf_counter()
            try:
                retrieved_docs = await search_one_question(
                    question=question,
                    source_config_id=source_config_id,
                    search_strategy=search_strategy,
                    strategy_config=strategy_config,
                    top_k=args.top_k,
                    multi_es_searcher=multi_es_searcher,
                )
            except Exception as e:
                # 单条失败不影响整批：记录空结果并继续
                logger.warning(f"问题 {idx + 1} 搜索失败: {e}")
                retrieved_docs = []
            query_time = time.perf_counter() - query_start
            print(f"  ✓ 检索到 {len(retrieved_docs)} 个结果，耗时: {query_time:.2f}s")
            return {
                "question_index": idx + 1,
                "question": question,
                "retrieved_docs": retrieved_docs,
            }

    tasks = [search_one(idx, item) for idx, item in enumerate(dataset)]
    results = await asyncio.gather(*tasks)

    # ── 保存结果 ──────────────────────────────────────────────────
    output_file = output_subdir / f"search_results_{args.dataset_name}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 同时保存一份为 search_results.json（方便查找最新）
    latest_file = output_subdir / "search_results.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("✅ 搜索完成!")
    print(f"  输出目录: {output_subdir}")
    print(f"  结果文件: {output_file}")
    print(f"  最新结果: {latest_file}")
    print(f"{'=' * 70}\n")

    if results:
        print("示例结果（第一个问题）:")
        print(json.dumps(results[0], ensure_ascii=False, indent=2)[:1000] + "...")

    return 0


if __name__ == "__main__":
    configure_logging()
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
