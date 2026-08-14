"""Shared async runner for Judge evaluation.

Manages semaphore, task creation, as_completed, single-sample failure isolation,
progress reporting, and token usage summary.
"""

import asyncio
import math
import time
from typing import Any

from pipeline.core.ai.base import BaseLLMClient
from pipeline.evaluation.judge.checkpoint import (
    atomic_write_json,
)
from pipeline.utils import get_logger

logger = get_logger(__name__)


async def run_judge_evaluation(
    samples: list[dict[str, Any]],
    evaluate_fn: Any,
    llm: BaseLLMClient,
    max_concurrent: int = 4,
    detailed_output: bool = True,
    checkpoint_path: str | None = None,
    only_metrics: list[str] | None = None,
    metrics: list[str] | None = None,
    **evaluate_kwargs: Any,
) -> dict[str, Any]:
    """Run Judge evaluation over all samples with concurrency control.

    Args:
        samples: List of prediction dicts with id, question, answer, etc.
        evaluate_fn: Async callable(sample, llm, metrics, detailed_output, **kwargs)
                     -> dict with "scores" and optional "llm_intermediate".
        llm: Pipeline LLM client.
        max_concurrent: Max concurrent LLM calls.
        detailed_output: Retained for compatibility; detailed output is always included.
        checkpoint_path: Path for intermittent checkpoint saves.
        only_metrics: Subset of metrics to evaluate (None = all).
        metrics: Full metric list for this evaluation type.
        **evaluate_kwargs: Extra kwargs passed to evaluate_fn.

    Returns:
        {"average_scores": {...}, "detailed": [...]} or dict of metric->average.
    """
    # Detailed results are mandatory for auditability; retain the keyword only
    # for compatibility with existing programmatic callers.
    detailed_output = True
    semaphore = asyncio.Semaphore(max_concurrent)
    total_tokens: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    async def _eval_one(i: int) -> dict[str, Any]:
        sample = samples[i]
        try:
            async with semaphore:
                result = await evaluate_fn(
                    sample=sample,
                    llm=llm,
                    metrics=metrics,
                    detailed_output=detailed_output,
                    **evaluate_kwargs,
                )
        except Exception as exc:
            logger.error(
                "Sample %s failed with %s",
                sample.get("id", "?"),
                type(exc).__name__,
            )
            failed_scores = {
                metric: float("nan") for metric in (metrics or [])
            }
            if detailed_output:
                return {
                    "id": sample.get("id"),
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("ground_truth", ""),
                    "generated_answer": sample.get(
                        "answer", sample.get("generated_answer", "")
                    ),
                    "contexts": sample.get(
                        "contexts", sample.get("context", [])
                    ),
                    "metrics": failed_scores,
                    "llm_intermediate": {},
                }
            return failed_scores
        if detailed_output:
            return {
                "id": sample["id"],
                "question": sample.get("question", ""),
                "ground_truth": sample.get("ground_truth", ""),
                "generated_answer": sample.get("answer", sample.get("generated_answer", "")),
                "contexts": sample.get("contexts", sample.get("context", [])),
                "metrics": result["scores"],
                "llm_intermediate": result.get("llm_intermediate"),
            }
        return result

    total = len(samples)
    tasks = [_eval_one(i) for i in range(total)]

    results: list[dict[str, Any]] = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            logger.error("Unexpected runner failure: %s", type(e).__name__)
            results.append({})
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            logger.info("Judge progress: %d/%d", completed, total)

        # Intermittent checkpoint save
        if checkpoint_path and completed % 5 == 0:
            try:
                ckpt = _build_summary(results, metrics or [], detailed_output)
                atomic_write_json(ckpt, checkpoint_path)
            except Exception as e:
                logger.warning("Checkpoint write skipped: %s", e)

    end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary = _build_summary(results, metrics or [], detailed_output)
    summary["total_tokens"] = total_tokens
    summary["start_time"] = start_time
    summary["end_time"] = end_time
    return summary


def _build_summary(
    results: list[dict[str, Any]],
    metric_names: list[str],
    detailed_output: bool,
) -> dict[str, Any]:
    """Compute per-metric averages from results list."""
    if detailed_output:
        all_metrics = set(metric_names)
        averages: dict[str, float] = {}
        for m in all_metrics:
            vals = [
                r["metrics"][m]
                for r in results
                if m in r.get("metrics", {})
                and not (
                    isinstance(r["metrics"].get(m), float)
                    and math.isnan(r["metrics"][m])
                )
            ]
            averages[m] = sum(vals) / len(vals) if vals else float("nan")
        return {"average_scores": averages, "detailed": results}

    # Non-detailed mode: results are flat metric->score dicts
    all_scores: dict[str, list[float]] = {}
    for r in results:
        for m, v in r.items():
            if isinstance(v, (int, float)):
                value = float(v)
                if not math.isnan(value):
                    all_scores.setdefault(m, []).append(value)
    averages = {
        m: sum(vals) / len(vals) if vals else float("nan")
        for m, vals in all_scores.items()
    }
    return averages
