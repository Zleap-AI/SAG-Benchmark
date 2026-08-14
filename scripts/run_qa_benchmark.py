#!/usr/bin/env python3
"""
QA Benchmark 脚本（检索增强问答 + 答案评估）

本脚本是检索（run_search_benchmark.py）之后的 **QA 步骤**：
    检索结果(retrieved_docs) → 拼 prompt → LLM 生成答案 → EM/F1 评分

设计依据：对齐 HippoRAG 2 (Zleap_SAG2) 的 QA 流程——
QA 阶段只消费「检索回来的 passage」(retrieved_docs, "title\\ncontent" 格式)，
取前 qa_top_k 条拼成 prompt，不依赖三元组/事实，与计算 Recall 用的是同一批 docs。

本脚本 **不重复实现检索**：输入必须是 run_search_benchmark.py 产出的
search_results.json（每条形如 {"question_index","question","retrieved_docs":[...]}），
检索相关的逻辑（strategy / source_config_id / ES 等）请直接用 run_search_benchmark.py。

输出：<output_dir>/qa_results.json 与 <output_dir>/run.log
    {
        "metrics": {"exact_match":..., "f1":...},
        "per_example": [{"question","predicted_answer","gold_answers","exact_match","f1"}, ...],
        "metadata": {...}
    }

示例：
    # 先跑检索
    uv run python scripts/run_search_benchmark.py --dataset-name sample --strategy sag2 --top-k 10
    # 再跑 QA，复用检索结果
    uv run python scripts/run_qa_benchmark.py --dataset-name sample \
        --input output/sample/sag2/<ts>/search_results.json --qa-top-k 5
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

from pipeline.core.ai.factory import create_llm_client
from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.core.config import get_settings
from pipeline.evaluation.qa_evaluator import QAEvaluator
from pipeline.evaluation.utils import (
    DatasetLoader,
    LLMTokenTracker,
    llm_tracking_scope,
    llm_tracking_stage,
)
from pipeline.utils import get_logger

logger = get_logger("scripts.run_qa_benchmark")

_QA_FILE_HANDLER_MARKER = "_run_qa_benchmark_file_handler"

# 压制噪声日志（与 run_search_benchmark.py 保持一致）
logging.getLogger("pipeline").setLevel(logging.WARNING)
logging.getLogger("pipeline.ai.llm").setLevel(logging.INFO)
logging.getLogger("pipeline.ai.openai").setLevel(logging.INFO)
for _noisy in ("httpx", "httpcore", "openai", "elasticsearch", "urllib3", "asyncio", "aiomysql"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _resolve_qa_output_dir(input_path: Path, explicit_output_dir: str | None, timestamp: str) -> Path:
    """Resolve this QA run's artifact directory before installing file logging."""

    if explicit_output_dir:
        return Path(explicit_output_dir)
    return input_path.parent / f"qa_{timestamp}"


def _qa_results_path(output_dir: Path) -> Path:
    return output_dir / "qa_results.json"


def _install_qa_file_handler(output_dir: Path) -> tuple[logging.FileHandler, Path]:
    """Install the only QA-owned root file handler for ``output_dir``.

    Reinitializing in one process removes and closes the previous QA handler;
    console handlers, non-QA file handlers, root level and logger levels remain
    untouched.
    """

    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, _QA_FILE_HANDLER_MARKER, False):
            root_logger.removeHandler(existing)
            existing.close()

    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    setattr(file_handler, _QA_FILE_HANDLER_MARKER, True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    return file_handler, log_file


# 对齐 HippoRAG 2 rag_qa 的系统提示（与 musique 模板同源：阅读理解 + Thought/Answer 输出）
QA_SYSTEM_PROMPT = (
    "You are an advanced reading comprehension assistant. Read the provided Wikipedia "
    "passages and answer the question at the end. Think briefly, then output your final "
    'answer on a new line prefixed with "Answer: ".'
)


def build_qa_prompt(question: str, retrieved_docs: list[str], qa_top_k: int) -> list[LLMMessage]:
    """
    构造 QA prompt，与 HippoRAG 2 (HippoRAG.qa) 对齐：
        user: "Wikipedia Title: {passage1}\n\n... \nQuestion: {q}\nThought: "
    retrieved_docs 每条为 "title\\ncontent"，作为 passage；截断到 qa_top_k。
    """
    passages = retrieved_docs[:qa_top_k]
    user_parts: list[str] = []
    for passage in passages:
        user_parts.append(f"Wikipedia Title: {passage}\n\n")
    user_parts.append(f"Question: {question}\nThought: ")

    return [
        LLMMessage(role=LLMRole.SYSTEM, content=QA_SYSTEM_PROMPT),
        LLMMessage(role=LLMRole.USER, content="".join(user_parts)),
    ]


def extract_answer(response_text: str) -> str:
    """
    从 LLM 响应中抽取答案。对齐 HippoRAG 2：按 "Answer:" 分隔，取最后一段并 trim。
    若未找到 "Answer:" 标记，则返回去首尾空白后的原文（兜底）。
    """
    if not response_text:
        return ""
    # 优先按 "Answer:" 分隔（大小写不敏感），取最后一段
    marker = "answer:"
    low = response_text.lower()
    idx = low.rfind(marker)
    if idx != -1:
        return response_text[idx + len(marker) :].strip()
    return response_text.strip()


def _load_search_results(input_path: Path) -> list[dict[str, Any]]:
    """加载 run_search_benchmark.py 产出的 search_results.json。"""
    if not input_path.exists():
        raise FileNotFoundError(f"检索结果文件不存在: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种字段命名：retrieved_docs（当前） / sections（旧版内存结构）
    normalized = []
    for item in data:
        docs = item.get("retrieved_docs") or item.get("sections") or []
        normalized.append(
            {
                "question_index": item.get("question_index"),
                "question": item.get("question", ""),
                "retrieved_docs": docs,
            }
        )
    return normalized


async def answer_one(
    client,
    question: str,
    retrieved_docs: list[str],
    qa_top_k: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, list[LLMMessage]]:
    """对单个问题调用 LLM 生成答案。

    返回 (抽取后的答案, LLM 原始响应文本, 构造的 prompt messages)。
    原始响应与 prompt 用于 --verbose 详细打印。
    """
    messages = build_qa_prompt(question, retrieved_docs, qa_top_k)
    async with semaphore:
        try:
            # 名为 show_retrieval_info：LLMTokenTracker.enable_llm_tracking 通过
            # 调用栈函数名识别 QA 阶段（见 token_tracker.py），此处适配其约定，
            # 使本次 QA 推理的 token 被正确归到 [QA] stage 而非 [UNKNOWN]。
            async def show_retrieval_info() -> Any:
                return await client.chat(messages=messages, temperature=0.0)

            response = await show_retrieval_info()
            return extract_answer(response.content), response.content, messages
        except Exception as e:
            logger.warning(f"问题「{question[:40]}...」QA 推理失败: {e}")
            return "", "", messages


async def run(
    args: argparse.Namespace,
    questions: list[str],
    retrieved_docs_list: list[list[str]],
    gold_answers: list[list[str]],
    timestamp: str,
    output_dir: Path,
    token_tracker: LLMTokenTracker,
) -> None:
    settings = get_settings()
    llm_model = settings.llm_model
    logger.info(f"📌 当前 LLM 模型: {llm_model}")

    # ── 创建 LLM 客户端 ───────────────────────────────────────
    logger.info("🤖 创建 QA LLM 客户端 ...")
    client = await create_llm_client(scenario="general")

    total = len(questions)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    logger.info(
        f"\n🚀 启动 QA 推理 ({total} 个问题, qa_top_k={args.qa_top_k}, 并发={args.max_concurrency})..."
    )
    logger.info("=" * 80)

    qa_start = time.perf_counter()
    # 并发推理，结果按 idx 对齐
    predictions: list[str | None] = [None] * total
    raw_responses: list[str] = [""] * total  # LLM 原始输出（--verbose 用）
    prompt_messages: list[list[LLMMessage] | None] = [None] * total  # prompt（--verbose 用）

    async def _run(idx: int) -> None:
        ans, raw, msgs = await answer_one(
            client, questions[idx], retrieved_docs_list[idx], args.qa_top_k, semaphore
        )
        predictions[idx] = ans
        raw_responses[idx] = raw
        prompt_messages[idx] = msgs
        if (idx + 1) % args.bench_size == 0 or (idx + 1) == total:
            logger.info(f"  QA 进度: {idx + 1}/{total}")

    await asyncio.gather(*(_run(i) for i in range(total)))
    qa_time = time.perf_counter() - qa_start

    final_predictions: list[str] = [p or "" for p in predictions]

    # ── 评估 EM / F1 ──────────────────────────────────────────
    qa_eval = QAEvaluator()
    per_example, pooled = qa_eval.evaluate_per_example(final_predictions, gold_answers)

    # ── 详细打印 prompt / 原始输出 / gold / EM·F1（--verbose）────
    if args.verbose:
        vlogger = logging.getLogger("scripts.run_qa_benchmark")
        for i in range(total):
            ga = gold_answers[i] if i < len(gold_answers) else []
            em = per_example[i]["exact_match"] if i < len(per_example) else None
            f1 = per_example[i]["f1"] if i < len(per_example) else None
            vlogger.info("\n" + "─" * 80)
            vlogger.info(f"【#{i + 1}】问题: {questions[i]}")

            msgs = prompt_messages[i]
            if msgs:
                for m in msgs:
                    vlogger.info(f"  ➤ prompt[{m.role.value}]:\n{m.content}")
            else:
                vlogger.info("  ➤ prompt: <构造失败/缺失>")
            vlogger.info(f"  ➤ LLM 原始输出:\n{raw_responses[i] or '<空>'}")
            vlogger.info(f"  ➤ 抽取后答案: {final_predictions[i]!r}")
            vlogger.info(f"  ➤ gold_answers: {ga}")
            vlogger.info(f"  ➤ 得分: EM={em}, F1={f1}")
            vlogger.info("─" * 80)

    # ── 打印汇总 ──────────────────────────────────────────────
    bench_logger = logging.getLogger("scripts.run_qa_benchmark")
    bench_logger.info("\n" + "=" * 80)
    bench_logger.info(f"📝 QA 评估结果: 数据集={args.dataset_name}, 共 {total} 个问题")
    bench_logger.info("=" * 50)
    bench_logger.info(
        f"  Exact Match: {pooled['exact_match']:.4f} ({pooled['exact_match']*100:.2f}%)"
    )
    bench_logger.info(f"  F1         : {pooled['f1']:.4f} ({pooled['f1']*100:.2f}%)")
    bench_logger.info("=" * 50)
    bench_logger.info(f"  QA 推理耗时: {qa_time:.1f} 秒")
    bench_logger.info("=" * 80)

    # ── 保存结果 ──────────────────────────────────────────────
    per_example_full = []
    for i in range(total):
        per_example_full.append(
            {
                "question_index": i + 1,
                "question": questions[i],
                "predicted_answer": final_predictions[i],
                "gold_answers": list(gold_answers[i]) if i < len(gold_answers) else [],
                "exact_match": per_example[i]["exact_match"] if i < len(per_example) else None,
                "f1": per_example[i]["f1"] if i < len(per_example) else None,
            }
        )

    qa_output = _qa_results_path(output_dir)
    # 提前算好 token 汇总，metadata 和日志末尾打印共用
    token_summary = token_tracker.get_summary()
    with open(qa_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": pooled,
                "per_example": per_example_full,
                "metadata": {
                    "dataset_name": args.dataset_name,
                    "llm_model": llm_model,
                    "qa_top_k": args.qa_top_k,
                    "max_concurrency": args.max_concurrency,
                    "total_questions": total,
                    "qa_time_seconds": round(qa_time, 2),
                    "input": str(args.input) if args.input else None,
                    "timestamp": timestamp,
                    "llm_token_usage": token_summary,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"💾 QA 结果已保存: {qa_output}")

    # ── token 统计（统一放在日志最后）────────────────────────
    bench_logger = logging.getLogger("scripts.run_qa_benchmark")
    bench_logger.info("\n" + "=" * 80)
    bench_logger.info("🔢 QA LLM Token 统计")
    bench_logger.info("=" * 50)
    bench_logger.info(f"  调用次数          : {token_summary['total_calls']}")
    bench_logger.info(f"  prompt tokens     : {token_summary['total_prompt']}")
    bench_logger.info(f"  completion tokens : {token_summary['total_completion']}")
    bench_logger.info(f"  total tokens      : {token_summary['total_tokens']}")
    for stage, s in token_summary["stages"].items():
        bench_logger.info(
            f"  [{stage}] calls={s['calls']}, "
            f"prompt={s['prompt']}, completion={s['completion']}, total={s['total']}"
        )
    bench_logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description="QA Benchmark：检索结果 → LLM 生成答案 → EM/F1 评估"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="数据集名称 (musique, hotpotqa, test_hotpotqa, sample 等)",
    )
    parser.add_argument(
        "--qa-top-k",
        type=int,
        default=5,
        help="QA prompt 中使用的前 K 条 retrieved_docs（默认 5，对齐 HippoRAG 2）",
    )
    parser.add_argument("--max-concurrency", type=int, default=4, help="QA 推理并发数（默认 4）")
    parser.add_argument(
        "--limit",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="限制处理范围：--limit N 只处理前N条；--limit S E 处理第S到第E条（0-based）",
    )
    parser.add_argument(
        "--bench-size", type=int, default=5, help="每 N 个问题打印一次进度（默认 5）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细打印每个问题的 QA prompt 模板、LLM 原始输出、抽取答案与 gold answers",
    )

    # 输入：run_search_benchmark.py 产出的检索结果（必填）
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="run_search_benchmark.py 产出的 search_results.json",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "本次 QA 的完整输出目录；qa_results.json 与 run.log 均写入此目录"
            "（默认：<search_results.json 同目录>/qa_<时间戳>/）"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_qa_output_dir(input_path, args.output_dir, timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only after the final output directory exists may QA attach its file log.
    _file_handler, log_file = _install_qa_file_handler(output_dir)
    logger.info(f"📝 日志将保存到: {log_file}")

    token_tracker = LLMTokenTracker()
    logger.info("✅ LLM token 追踪已启用（当前运行上下文）")

    # ── 解析 --limit ──────────────────────────────────────────
    limit_start: int | None = None
    limit_end: int | None = None
    if args.limit:
        if len(args.limit) == 1:
            limit_end = args.limit[0]
        elif len(args.limit) == 2:
            limit_start, end_incl = args.limit
            if limit_start > end_incl:
                logger.error(f"--limit 起始索引 {limit_start} > 结束索引 {end_incl}")
                sys.exit(1)
            limit_end = end_incl + 1
        else:
            logger.error("--limit 最多接受两个整数")
            sys.exit(1)

    # ── 加载数据集（问题 + gold_answers）─────────────────────
    logger.info(f"Loading dataset: {args.dataset_name}")
    loader = DatasetLoader(args.dataset_name)
    questions_all = loader.get_questions()
    gold_answers_all = loader.get_gold_answers()
    gold_answers_all = [list(ga) for ga in gold_answers_all]  # set → list

    if limit_start is not None or limit_end is not None:
        s = limit_start or 0
        e = limit_end if limit_end is not None else len(questions_all)
        questions_all = questions_all[s:e]
        gold_answers_all = gold_answers_all[s:e]

    # ── 加载检索结果（来自 run_search_benchmark.py）──────────────
    logger.info(f"📥 复用检索结果: {args.input}")
    search_results = _load_search_results(Path(args.input))
    # 与 search_results 按 idx 同步裁剪（若指定了 --limit）
    if limit_start is not None or limit_end is not None:
        s = limit_start or 0
        e = limit_end if limit_end is not None else len(search_results)
        search_results = search_results[s:e]
    questions = [r["question"] for r in search_results]
    retrieved_docs_list = [r["retrieved_docs"] for r in search_results]
    # 若 search_results 自带 question，优先用之（保证与 retrieved_docs 对齐）；
    # gold_answers 仍来自数据集，按相同 idx 裁剪。
    gold_answers = gold_answers_all[: len(questions)]

    if len(questions) != len(gold_answers):
        logger.warning(
            f"questions({len(questions)}) 与 gold_answers({len(gold_answers)}) 长度不一致，"
            f"按较短者对齐"
        )
        n = min(len(questions), len(gold_answers))
        questions = questions[:n]
        gold_answers = gold_answers[:n]
        retrieved_docs_list = retrieved_docs_list[:n]

    logger.info(f"❓ 待评估问题数: {len(questions)}")
    logger.info(f"  输出目录: {output_dir}\n" + "=" * 80)

    with llm_tracking_scope(token_tracker), llm_tracking_stage("QA"):
        await run(
            args, questions, retrieved_docs_list, gold_answers, timestamp, output_dir, token_tracker
        )
    logger.info("\n✅ QA Benchmark 完成！")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
