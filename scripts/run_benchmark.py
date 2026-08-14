#!/usr/bin/env python3
"""
交叉验证脚本：计算召回结果的 Recall@K 指标

用法:
    # 自动查找最新结果（推荐）
    python scripts/run_benchmark.py \
        --dataset musique \
        --k-list 1 2 5 10

    # 手动指定结果文件（如果需要评估特定版本的结果）
    python scripts/run_benchmark.py \
        --results pipeline/evaluation/outputs/SAG/qwen3.6-35b-a3b/musique/20260516_195254/results_musique.json \
        --dataset musique \
        --k-list 1 2 5 10
python scripts/run_benchmark.py \
        --results pipeline/evaluation/outputs/SAG/qwen3.6-35b-a3b/musique/20260516_195254/results_musique.json \
        --dataset musique \
        --k-list 1 2 5 10

功能:
    1. 自动查找或手动加载召回结果文件 (results_*.json)
    2. 加载对应数据集的标准答案 (gold_docs)
    3. 计算 Recall@1, @2, @5, @10
    4. 输出详细的统计信息
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.evaluation.metrics import RetrievalRecall
from pipeline.evaluation.utils import DatasetLoader
from pipeline.utils import get_logger

logger = get_logger("scripts.run_benchmark")


@dataclass
class RecallStatistics:
    """召回统计信息"""

    total_questions: int
    full_recall_count: int  # 全部召回
    partial_recall_count: int  # 部分召回
    zero_recall_count: int  # 零召回
    recall_metrics: dict[str, float]  # Recall@K 指标
    zero_recall_details: list[dict[str, Any]] = None  # 0 召回题目详情


def find_latest_results(dataset_name: str, base_dir: Path = None) -> Path:
    """
    查找指定数据集的最新结果文件

    Args:
        dataset_name: 数据集名称 (hotpotqa, musique, etc.)
        base_dir: 基础目录，默认为 pipeline/evaluation/outputs

    Returns:
        最新结果文件的路径
    """
    if base_dir is None:
        base_dir = project_root / "pipeline" / "evaluation" / "outputs"

    # 查找所有匹配的结果文件
    pattern = f"**/results_{dataset_name}.json"
    result_files = list(base_dir.glob(pattern))

    if not result_files:
        raise FileNotFoundError(f"未找到数据集 '{dataset_name}' 的结果文件")

    # 按修改时间排序，返回最新的
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def load_retrieval_results(results_path: str) -> list[dict[str, Any]]:
    """
    加载召回结果文件

    Args:
        results_path: results_*.json 文件路径

    Returns:
        召回结果列表
    """
    logger.info(f"📂 加载召回结果: {results_path}")
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    logger.info(f"✅ 成功加载 {len(results)} 个问题的召回结果\n")
    return results


def load_gold_docs(dataset_name: str) -> list[list[str]]:
    """
    加载数据集的标准答案文档（直接从 JSON 文件解析）

    Args:
        dataset_name: 数据集名称 (hotpotqa, musique, etc.)

    Returns:
        标准答案文档列表，格式为 ["title\ncontent", ...]
    """
    logger.info(f"📚 加载数据集标准答案: {dataset_name}")

    # 直接读取数据集 JSON 文件
    dataset_path = project_root / "pipeline" / "evaluation" / "dataset" / f"{dataset_name}.json"
    if not dataset_path.exists():
        # fallback: 兼容旧版路径
        dataset_path = project_root / "pipeline" / "evaluation" / "dataset" / f"{dataset_name}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    gold_docs_list = []

    # 根据数据集类型处理
    if dataset_name in ["hotpotqa", "test_hotpotqa"]:
        # HotpotQA 格式: supporting_facts + context
        # 与 DatasetLoader.get_gold_docs_for_recall() 保持一致
        for item in dataset:
            supporting_facts = item.get("supporting_facts", [])
            context = item.get("context", [])

            # 提取 supporting_facts 涉及的 title
            gold_title = {title for title, _ in supporting_facts}

            # 构建金标准文档: "title\ncontent"（content 用 ''.join() 直接拼接）
            gold_docs = []
            for title, sentences in context:
                if title in gold_title:
                    # 与 DatasetLoader 一致：hotpotqa/test_hotpotqa 用 ''.join()
                    content = "".join(sentences)
                    gold_docs.append(f"{title}\n{content}")

            gold_docs_list.append(gold_docs)

    elif dataset_name == "2wikimultihopqa":
        # 2WikiMultihopQA 格式: supporting_facts + context
        # 需要匹配 title + content（用 ' '.join() 拼接句子）
        for item in dataset:
            supporting_facts = item.get("supporting_facts", [])
            context = item.get("context", [])

            # 提取 supporting_facts 涉及的 title
            gold_title = {title for title, _ in supporting_facts}

            # 构建金标准文档: "title\ncontent"（content 用 ' '.join() 拼接）
            gold_docs = []
            for title, sentences in context:
                if title in gold_title:
                    content = " ".join(sentences)
                    gold_docs.append(f"{title}\n{content}")

            gold_docs_list.append(gold_docs)

    elif dataset_name == "musique":
        # Musique 格式: paragraphs 中的 is_supporting 标记
        # 需要匹配 title + paragraph_text
        for item in dataset:
            gold_docs = []
            for para in item.get("paragraphs", []):
                if para.get("is_supporting", False):
                    title = para.get("title", "")
                    content = para.get("paragraph_text", "")
                    gold_docs.append(f"{title}\n{content}")
            gold_docs_list.append(gold_docs)

    else:
        # 其他数据集使用 DatasetLoader
        loader = DatasetLoader(dataset_name)
        gold_docs_list = loader.get_gold_docs_for_recall(limit=None)

    logger.info(f"✅ 成功加载 {len(gold_docs_list)} 个问题的标准答案\n")
    return gold_docs_list


def load_questions(dataset_name: str) -> list[str]:
    """
    加载数据集的提问（按原始顺序，与 load_gold_docs 的 gold_docs_list 逐题对齐）。

    Args:
        dataset_name: 数据集名称 (hotpotqa, musique, etc.)

    Returns:
        提问列表，与 gold_docs 同序、同长
    """
    dataset_path = project_root / "pipeline" / "evaluation" / "dataset" / f"{dataset_name}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = [item.get("question", "") for item in dataset]
    return questions


def calculate_recall_metrics(
    retrieval_results: list[dict[str, Any]],
    gold_docs: list[list[str]],
    dataset_name: str,
    k_list: list[int] = None,
    questions: list[str] = None,
) -> RecallStatistics:
    """
    计算 Recall@K 指标和召回统计

    Args:
        retrieval_results: 召回结果列表
        gold_docs: 标准答案文档列表
        dataset_name: 数据集名称（用于确定匹配策略）
        k_list: K 值列表
        questions: 提问列表（与 gold_docs 同序），用于记录 0 召回详情

    Returns:
        RecallStatistics 对象
    """
    if k_list is None:
        k_list = [1, 2, 5, 10]
    logger.info("🔍 开始计算 Recall 指标...")
    logger.info(f"   数据集: {dataset_name}")
    logger.info(f"   K 值列表: {k_list}\n")

    # 准备数据格式
    retrieved_docs_list = []
    for result in retrieval_results:
        # retrieved_docs 是段落文本列表
        retrieved_docs_list.append(result["retrieved_docs"])

    # 确保数据长度一致
    min_len = min(len(retrieved_docs_list), len(gold_docs))
    if len(retrieved_docs_list) != len(gold_docs):
        logger.warning(
            f"⚠️  警告: 召回结果数 ({len(retrieved_docs_list)}) 与标准答案数 ({len(gold_docs)}) 不一致"
        )
        logger.warning(f"   将使用前 {min_len} 个问题进行计算\n")
        retrieved_docs_list = retrieved_docs_list[:min_len]
        gold_docs = gold_docs[:min_len]

    # 统一匹配策略：所有数据集都使用完整的 "title\ncontent" 字符串匹配
    # 这与 benchmark.py 的 _match_retrieved_docs() 方法保持一致
    logger.info("   匹配策略: 完整字符串匹配 title\\ncontent（与 benchmark.py 一致）\n")

    # 使用 RetrievalRecall 计算指标
    recall_metric = RetrievalRecall()
    pooled_recall, example_recalls = recall_metric.calculate_metric_scores(
        gold_docs=gold_docs, retrieved_docs=retrieved_docs_list, k_list=k_list
    )

    # 统计召回情况（与 benchmark.py 的 _handle_bench_callback 方法一致）
    full_count = 0
    partial_count = 0
    zero_count = 0
    zero_recall_details: list[dict[str, Any]] = []

    # questions 对齐到 min_len（与 retrieved/gold 截断一致）
    aligned_questions = questions or []
    if len(aligned_questions) > min_len:
        aligned_questions = aligned_questions[:min_len]

    for i, (retrieved, gold) in enumerate(zip(retrieved_docs_list, gold_docs)):
        # 直接字符串匹配（与 benchmark.py 第 632 行一致）
        matched = [doc for doc in retrieved if doc in gold]

        if len(matched) == 0:
            zero_count += 1
            question = aligned_questions[i] if i < len(aligned_questions) else ""
            zero_recall_details.append(
                {
                    "index": i,
                    "question": question,
                    "gold_docs": gold,  # supporting paragraphs (标准答案文档)
                    "retrieved_docs": retrieved,  # 检索返回的内容
                    "num_retrieved": len(retrieved),
                    "num_gold": len(gold),
                }
            )
        elif len(matched) < len(gold):
            partial_count += 1
        else:
            full_count += 1

    return RecallStatistics(
        total_questions=len(retrieved_docs_list),
        full_recall_count=full_count,
        partial_recall_count=partial_count,
        zero_recall_count=zero_count,
        recall_metrics=pooled_recall,
        zero_recall_details=zero_recall_details,
    )


def print_statistics(stats: RecallStatistics) -> None:
    """
    打印统计结果

    Args:
        stats: RecallStatistics 对象
    """
    logger.info("=" * 80)
    logger.info("📊 召回统计结果")
    logger.info("=" * 80)

    # 基本统计
    logger.info(f"\n总问题数: {stats.total_questions}")
    logger.info(
        f"✅ 全部召回: {stats.full_recall_count} 个 ({stats.full_recall_count/stats.total_questions*100:.2f}%)"
    )
    logger.warning(
        f"⚠️  部分召回: {stats.partial_recall_count} 个 ({stats.partial_recall_count/stats.total_questions*100:.2f}%)"
    )
    logger.error(
        f"❌ 零召回: {stats.zero_recall_count} 个 ({stats.zero_recall_count/stats.total_questions*100:.2f}%)"
    )

    # Recall@K 指标（按 K 值从小到大排序）
    logger.info("\n📈 Recall@K 指标:")
    logger.info("-" * 50)
    # 提取 K 值并排序
    metrics_with_k = [
        (int(metric.split("@")[1]), metric, score) for metric, score in stats.recall_metrics.items()
    ]
    metrics_with_k.sort(key=lambda x: x[0])  # 按 K 值排序

    for k_val, _metric, score in metrics_with_k:
        logger.info(f"  Recall@{k_val:>2}: {score:.4f} ({score*100:.2f}%)")

    logger.info("=" * 80)


def print_zero_recall_details(stats: RecallStatistics) -> None:
    """
    打印 0 召回题目的详情：提问、标准答案文档(supporting paragraphs)、检索返回内容。

    Args:
        stats: RecallStatistics 对象（含 zero_recall_details）
    """
    details = stats.zero_recall_details or []
    if not details:
        logger.info("\n✅ 无 0 召回题目\n")
        return

    logger.info("\n" + "=" * 80)
    logger.error(f"❌ 0 召回题目详情（共 {len(details)} 个）")
    logger.info("=" * 80)

    for idx, d in enumerate(details, 1):
        logger.info(f"\n----- [0召回 #{idx}] 原始索引={d['index']} -----")
        logger.info(f"【提问 question】\n{d['question']}")
        logger.info(f"\n【标准答案 supporting paragraphs】(gold_docs, 共 {d['num_gold']} 个)")
        for j, gold in enumerate(d["gold_docs"], 1):
            # supporting paragraph 格式为 "title\ncontent"，分行展示更清晰
            parts = gold.split("\n", 1)
            title = parts[0] if parts else ""
            content = parts[1] if len(parts) > 1 else ""
            logger.info(f"  ({j}) [{title}] {content}")
        logger.info(f"\n【检索返回 retrieved_docs】(共 {d['num_retrieved']} 个)")
        if not d["retrieved_docs"]:
            logger.info("  (空 — 检索未返回任何文档)")
        for j, ret in enumerate(d["retrieved_docs"], 1):
            parts = ret.split("\n", 1)
            title = parts[0] if parts else ""
            content = parts[1] if len(parts) > 1 else ret
            logger.info(f"  ({j}) [{title}] {content}")

    logger.info("\n" + "=" * 80)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import argparse

    parser = argparse.ArgumentParser(
        description="计算召回结果的 Recall@K 指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动查找最新结果（推荐）
  python scripts/run_benchmark.py \\
      --dataset musique \\
      --k-list 1 2 5 10

  # 手动指定结果文件（如果需要评估特定版本的结果）
  python scripts/run_benchmark.py \\
      --results pipeline/evaluation/outputs/SAG/qwen3.6-35b-a3b/musique/20260516_195254/results_musique.json \\
      --dataset musique \\
      --k-list 1 2 5 10
        """,
    )

    parser.add_argument(
        "--results", type=str, help="召回结果文件路径 (results_*.json)，不指定则自动查找最新结果"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpotqa", "musique", "2wikimultihopqa", "test_hotpotqa"],
        help="数据集名称",
    )

    parser.add_argument(
        "--k-list", type=int, nargs="+", default=[1, 2, 5, 10], help="K 值列表 (默认: 1 2 5 10)"
    )

    args = parser.parse_args()

    # 确定召回结果文件路径
    if args.results:
        # 使用用户指定的结果文件
        results_path = args.results
        if not Path(results_path).exists():
            logger.error(f"❌ 错误: 文件不存在: {results_path}")
            sys.exit(1)
    else:
        # 自动查找最新结果（默认行为）
        try:
            latest_file = find_latest_results(args.dataset)
            results_path = str(latest_file)
            logger.info(f"🔍 自动找到最新结果: {results_path}\n")
        except FileNotFoundError as e:
            logger.error(f"❌ 错误: {e}")
            logger.error("💡 提示: 请使用 --results 参数手动指定结果文件路径")
            sys.exit(1)

    try:
        # 1. 加载召回结果
        retrieval_results = load_retrieval_results(results_path)

        # 2. 加载标准答案
        gold_docs = load_gold_docs(args.dataset)

        # 2.1 加载提问（与 gold_docs 同序，用于 0 召回详情打印）
        questions = load_questions(args.dataset)

        # 3. 计算指标
        stats = calculate_recall_metrics(
            retrieval_results=retrieval_results,
            gold_docs=gold_docs,
            dataset_name=args.dataset,
            k_list=args.k_list,
            questions=questions,
        )

        # 4. 打印结果
        print_statistics(stats)

        # 5. 打印 0 召回题目详情（提问 + 标准答案 + 检索返回）
        #print_zero_recall_details(stats)

    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
