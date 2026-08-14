"""Generation Judge evaluation — migrated from external/judge/Evaluation/generation_eval.py.

Evaluates generation quality metrics: EM, F1, ROUGE, answer_correctness,
and coverage_score. No embedding/datasets/LangChain dependency.
"""

import asyncio
import json
import math
import os
from typing import Any

from pipeline.evaluation.judge.checkpoint import (
    _select_rerun_ids,
    atomic_write_json,
    load_checkpoint,
    merge_partial_results,
)
from pipeline.evaluation.judge.metrics import (
    answer_correctness,
    coverage,
    rouge,
)
from pipeline.evaluation.utils.eval_utils import normalize_answer
from pipeline.utils import get_logger

logger = get_logger(__name__)

ALL_GENERATION_METRICS = {
    "qa_em",
    "qa_f1",
    "rouge_score",
    "answer_correctness",
    "coverage_score",
}


def _is_graph_based_context(context: str) -> bool:
    """Return whether context uses the structured graph-section format."""
    return isinstance(context, str) and context.strip().startswith("-----")


def _preprocess_context(
    context: str | list[str],
    context_top_k: int = 5,
) -> list[str]:
    """Preserve graph context and limit plain passage context to top-k chunks."""
    if isinstance(context, list):
        context = "\n\n".join(str(item) for item in context if item)
    if not isinstance(context, str) or not context.strip():
        return []
    if _is_graph_based_context(context) or context_top_k <= 0:
        return [context]
    chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    return chunks[:context_top_k]


def _average_scores(scores: dict[str, list[float]]) -> dict[str, float]:
    """Average metric values while excluding failed-sample NaNs."""
    averages: dict[str, float] = {}
    for metric, values in scores.items():
        valid = [
            value
            for value in values
            if not (isinstance(value, float) and math.isnan(value))
        ]
        averages[metric] = sum(valid) / len(valid) if valid else float("nan")
    return averages


def _canonical_gt_text(ground_truth: str | list[str]) -> str:
    """Return first non-empty gold for metrics requiring a single string."""
    if isinstance(ground_truth, list):
        for answer in ground_truth:
            if isinstance(answer, str) and answer.strip():
                return answer
        return ""
    return ground_truth if isinstance(ground_truth, str) else ""


def _compute_qa_em(answer: str, ground_truth: str | list[str]) -> float:
    """MRQA-style Exact Match."""
    golds = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    norm_ans = normalize_answer(answer) if answer.strip() else ""
    for gold in golds:
        if not gold.strip() or not answer.strip():
            continue
        if norm_ans == normalize_answer(gold):
            return 1.0
    return 0.0


def _compute_qa_f1(answer: str, ground_truth: str | list[str]) -> float:
    """MRQA-style token-level F1."""
    from collections import Counter

    golds = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    max_f1 = 0.0
    for gold in golds:
        if not gold.strip() or not answer.strip():
            continue
        pred_tokens = normalize_answer(answer).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


async def evaluate_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | list[str],
    metrics: list[str],
    llm: Any,
    detailed_output: bool = True,
) -> dict[str, Any]:
    """Evaluate generation metrics for a single sample."""
    detailed_output = True
    results: dict[str, float] = {}
    intermediates: dict[str, Any] = {} if detailed_output else {}
    gt_text = _canonical_gt_text(ground_truth)

    coros = {}
    if "rouge_score" in metrics:
        coros["rouge_score"] = rouge.compute_rouge_score(answer, gt_text)

    async def _wrap_sync(fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    if "qa_f1" in metrics:
        coros["qa_f1"] = _wrap_sync(_compute_qa_f1, answer, ground_truth)

    if "qa_em" in metrics:
        coros["qa_em"] = _wrap_sync(_compute_qa_em, answer, ground_truth)

    if "answer_correctness" in metrics:
        coros["answer_correctness"] = answer_correctness.compute_answer_correctness(
            question, answer, gt_text, llm, return_intermediate=detailed_output
        )

    if "coverage_score" in metrics:
        coros["coverage_score"] = coverage.compute_coverage_score(
            question, gt_text, answer, llm, return_intermediate=detailed_output
        )

    gathered = await asyncio.gather(*coros.values())

    _llm_metrics = {"coverage_score", "answer_correctness"}
    for i, metric in enumerate(coros.keys()):
        result = gathered[i]
        if detailed_output and metric in _llm_metrics:
            score_val, intermediate = result
            results[metric] = score_val
            intermediates[metric] = intermediate  # type: ignore[index]
        else:
            results[metric] = result

    if detailed_output:
        return {"scores": results, "llm_intermediate": intermediates}
    return results


async def run_generation_eval(
    data_file: str,
    llm: Any,
    output_file: str | None = None,
    detailed_output: bool = True,
    max_concurrent: int = 4,
    only_metrics: list[str] | None = None,
    force: bool = False,
    dataset: str | None = None,
    dataset_dir: str = "dataset",
    num_samples: int | None = None,
    context_top_k: int = 5,
) -> dict[str, Any]:
    """Run full generation evaluation pipeline."""
    detailed_output = True
    with open(data_file, encoding="utf-8") as f:
        file_data = json.load(f)

    # Determine metrics to evaluate
    eval_metrics = only_metrics if only_metrics else list(ALL_GENERATION_METRICS)
    if only_metrics:
        illegal = [m for m in only_metrics if m not in ALL_GENERATION_METRICS]
        if illegal:
            logger.error("Invalid metrics: %s", illegal)
            raise SystemExit(1)

    # Output path
    judge_model = getattr(getattr(llm, "config", None), "model", "deterministic")
    out_dir = os.path.dirname(output_file) if output_file else "."
    out_name = os.path.basename(output_file) if output_file else "results.json"
    judge_dir = os.path.join(out_dir, f"LlmJudge_{judge_model}")
    final_path = os.path.join(judge_dir, out_name)
    ckpt_path = final_path + ".partial"

    # Resume from checkpoint
    existing: dict[str, Any] = {}
    resume_mode = False
    if not force and os.path.exists(final_path):
        existing = load_checkpoint(final_path) or {}
        if existing:
            resume_mode = True
            logger.info("Found existing results, resuming (NaN/missing only)")

    # Group by question type
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        grouped.setdefault(q_type, []).append(item)

    all_results: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    for question_type, group_items in grouped.items():
        logger.info("Evaluating: %s (%d samples)", question_type, len(group_items))
        if num_samples is not None:
            group_items = group_items[:num_samples]

        # Resume logic
        rerun_ids: set[int] | None = None
        if resume_mode and question_type in existing:
            old_detailed = existing[question_type].get("detailed", [])
            all_ids_q = {it["id"] for it in group_items}
            rerun_ids = _select_rerun_ids(old_detailed, all_ids_q, force_all=bool(only_metrics))
            if rerun_ids:
                group_items = [it for it in group_items if it["id"] in rerun_ids]
                logger.info("  Re-evaluating %d samples", len(rerun_ids))
            else:
                logger.info("  All complete, skipping")
                all_results[question_type] = existing[question_type]
                continue

        # Evaluate
        detailed_entries: list[dict[str, Any]] = []
        all_scores: dict[str, list[float]] = {}

        async def _eval_one(item: dict[str, Any]) -> dict[str, Any]:
            try:
                async with semaphore:
                    ctx = _preprocess_context(
                        item.get("contexts", item.get("context", [])),
                        context_top_k,
                    )
                    result = await evaluate_sample(
                        question=item["question"],
                        answer=item.get("answer", item.get("generated_answer", "")),
                        contexts=ctx,
                        ground_truth=item.get("ground_truth", ""),
                        metrics=eval_metrics,
                        llm=llm,
                        detailed_output=detailed_output,
                    )
                return {"id": item["id"], "item": item, "result": result}
            except Exception as exc:
                logger.error(
                    "Sample %s failed with %s",
                    item.get("id", "?"),
                    type(exc).__name__,
                )
                return {"id": item.get("id"), "item": item, "error": exc}

        tasks = [_eval_one(it) for it in group_items]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            entry = await coro
            item = entry["item"]
            if "error" in entry:
                failed_scores = {metric: float("nan") for metric in eval_metrics}
                result = (
                    {"scores": failed_scores, "llm_intermediate": {}}
                    if detailed_output
                    else failed_scores
                )
            else:
                result = entry["result"]
            completed += 1

            if detailed_output:
                detailed_entries.append({
                    "id": item["id"],
                    "question": item.get("question", ""),
                    "ground_truth": item.get("ground_truth", ""),
                    "generated_answer": item.get("answer", item.get("generated_answer", "")),
                    "contexts": item.get("contexts", item.get("context", [])),
                    "metrics": result["scores"],
                    "llm_intermediate": result.get("llm_intermediate"),
                })
                for m, v in result["scores"].items():
                    all_scores.setdefault(m, []).append(v)
            else:
                for m, v in result.items():
                    all_scores.setdefault(m, []).append(v)

            if completed % max(1, len(group_items) // 10) == 0:
                logger.info("  Progress: %d/%d", completed, len(group_items))

            # Intermittent checkpoint
            if completed % 5 == 0:
                try:
                    avg = _average_scores(all_scores)
                    if existing and question_type in existing:
                        ckpt_data = dict(existing)
                        ckpt_data[question_type] = {
                            "average_scores": avg,
                            "detailed": detailed_entries if detailed_output else [],
                        }
                    else:
                        ckpt_data = {
                            question_type: {
                                "average_scores": avg,
                                "detailed": detailed_entries if detailed_output else [],
                            }
                        }
                    atomic_write_json(ckpt_data, ckpt_path)
                except Exception as e:
                    logger.warning("Checkpoint write skipped: %s", e)

        # Compute averages
        avg = _average_scores(all_scores)
        q_result = {"average_scores": avg, "detailed": detailed_entries} if detailed_output else avg

        # Merge with existing
        if resume_mode and question_type in existing and rerun_ids:
            q_result = merge_partial_results(
                existing, {question_type: q_result}, question_type, rerun_ids, only_metrics
            )

        all_results[question_type] = q_result

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Save final
    if output_file:
        os.makedirs(judge_dir, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", final_path)

    return all_results
