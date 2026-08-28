"""Shared async runner for Judge evaluation.

Provides SampleEvaluator Protocol and JudgeEvaluationRunner as the single
execution kernel for generation and retrieval evaluation.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Protocol

from pipeline.core.ai.base import BaseLLMClient
from pipeline.evaluation.judge.checkpoint import atomic_write_json, load_checkpoint
from pipeline.evaluation.judge.errors import (
    JudgeExecutionError,
    MetricResultError,
)
from pipeline.evaluation.judge.metric_validation import validate_metric_mapping
from pipeline.evaluation.judge.models import (
    EvaluationKind,
    JudgeDetailedResult,
    JudgeRunSummary,
    JudgeSample,
    SampleEvaluationStatus,
)
from pipeline.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from pipeline.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected failure types that don't warrant aborting the entire run
# ---------------------------------------------------------------------------

_EXPECTED_FAILURE_TYPES = (
    asyncio.TimeoutError,
    TimeoutError,
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    ConnectionError,
)


# ---------------------------------------------------------------------------
# SampleEvaluator Protocol
# ---------------------------------------------------------------------------


class SampleEvaluator(Protocol):
    """Protocol for per-sample evaluation (generation or retrieval)."""

    kind: EvaluationKind
    supported_metrics: tuple[str, ...]
    default_metrics: tuple[str, ...]

    async def evaluate(
        self,
        sample: JudgeSample,
        llm: BaseLLMClient | None,
        metrics: tuple[str, ...],
        context_top_k: int,
    ) -> JudgeDetailedResult: ...


# ---------------------------------------------------------------------------
# JudgeEvaluationRunner
# ---------------------------------------------------------------------------


class JudgeEvaluationRunner:
    """Single execution kernel for Judge evaluation.

    Manages concurrency, checkpointing, progress reporting, and
    failure isolation: expected LLM failures (timeout, connection) are
    recorded per-sample; programming errors cancel the entire run.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        checkpoint_interval: int = 5,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._checkpoint_interval = checkpoint_interval

    async def run(
        self,
        samples: list[JudgeSample],
        evaluator: SampleEvaluator,
        llm: BaseLLMClient | None,
        metrics: tuple[str, ...],
        context_top_k: int = 5,
        checkpoint_path: str | None = None,
        retry_failed: bool = False,
    ) -> JudgeRunSummary:
        """Execute evaluation over all samples and return summary."""
        semaphore = asyncio.Semaphore(self._max_concurrent)
        start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        total = len(samples)
        met_names = list(metrics)

        detailed_results: list[JudgeDetailedResult] = []
        all_scores: dict[str, list[float]] = {metric: [] for metric in met_names}
        completed_keys: set[int] = set()
        if checkpoint_path:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                detailed_results, completed_keys = _restore_checkpoint(
                    checkpoint, samples, retry_failed=retry_failed
                )
                for result in detailed_results:
                    for metric, value in result.metrics.items():
                        all_scores.setdefault(metric, []).append(value)
                logger.info(
                    "Resuming Judge checkpoint: %d/%d samples already completed",
                    len(detailed_results),
                    len(samples),
                )
        programming_error: Exception | None = None

        async def _eval_one(sample: JudgeSample) -> JudgeDetailedResult:
            nonlocal programming_error
            try:
                async with semaphore:
                    result = await evaluator.evaluate(
                        sample=sample,
                        llm=llm,
                        metrics=metrics,
                        context_top_k=context_top_k,
                    )
                    validate_metric_mapping(
                        result.metrics,
                        scope=f"sample id={sample.id} seq={sample.seq}",
                    )
                    return result
            except MetricResultError:
                # Metric integrity errors are fatal: do not turn them into a
                # NaN placeholder that could be persisted as a valid run.
                raise
            except _EXPECTED_FAILURE_TYPES as exc:
                logger.error(
                    "Sample %s failed with expected %s: %s",
                    sample.id,
                    type(exc).__name__,
                    exc,
                )
                return JudgeDetailedResult(
                    id=sample.id,
                    seq=sample.seq,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    generated_answer=sample.answer,
                    contexts=sample.contexts,
                    metrics={},
                    llm_intermediate={},
                    status=SampleEvaluationStatus.SAMPLE_FAILED,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
            except Exception as exc:
                # Programming errors: cancel remaining work, fail the run
                logger.error(
                    "Sample %s hit programming error %s: %s — aborting run",
                    sample.id,
                    type(exc).__name__,
                    exc,
                )
                programming_error = exc
                return JudgeDetailedResult(
                    id=sample.id,
                    seq=sample.seq,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    generated_answer=sample.answer,
                    contexts=sample.contexts,
                    metrics={},
                    llm_intermediate={},
                    status=SampleEvaluationStatus.PROGRAMMING_ERROR,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )

        tasks = [
            asyncio.create_task(_eval_one(sample))
            for sample in samples
            if not completed_keys or _sample_key(sample) not in completed_keys
        ]
        completed = len(detailed_results)
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except Exception as exc:
                # coroutine-level error: cancel all pending tasks
                for t in tasks:
                    t.cancel()
                logger.error("Fatal runner error: %s", exc)
                raise JudgeExecutionError(
                    f"Judge run aborted due to unhandled error: {type(exc).__name__}: {exc}"
                ) from exc
            detailed_results.append(result)
            completed += 1

            for m, v in result.metrics.items():
                all_scores.setdefault(m, []).append(v)

            if completed % max(1, total // 10) == 0 or completed == total:
                logger.info("Judge progress: %d/%d", completed, total)

            if checkpoint_path and (retry_failed or completed % self._checkpoint_interval == 0):
                try:
                    avg = _compute_averages(all_scores)
                    ckpt = {
                        "average_scores": avg,
                        "detailed": [r.model_dump() for r in detailed_results],
                    }
                    atomic_write_json(ckpt, checkpoint_path)
                except Exception as exc:
                    logger.warning("Checkpoint write skipped: %s", exc)

            # If a programming error was recorded, cancel remaining tasks
            if programming_error is not None:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

        end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        avg = _compute_averages(all_scores)

        successful = sum(1 for r in detailed_results if r.status == SampleEvaluationStatus.SUCCESS)
        failed = len(detailed_results) - successful

        valid_counts: dict[str, int] = {}
        for m, vals in all_scores.items():
            valid_counts[m] = sum(1 for v in vals if not (isinstance(v, float) and math.isnan(v)))

        if programming_error is not None:
            raise JudgeExecutionError(
                f"Programming error in sample evaluation: "
                f"{type(programming_error).__name__}: {programming_error}"
            )

        # A failed sample deliberately carries no metrics. Every metric that
        # remains in a completed result or aggregate must be finite.
        try:
            for result in detailed_results:
                validate_metric_mapping(
                    result.metrics,
                    scope=f"sample id={result.id} seq={result.seq}",
                )
            validate_metric_mapping(avg, scope="run average_scores")
        except MetricResultError as exc:
            raise JudgeExecutionError(f"Metric validation failed: {exc}") from exc

        return JudgeRunSummary(
            average_scores=avg,
            detailed=detailed_results,
            total_tokens={},
            start_time=start_time,
            end_time=end_time,
            total_samples=total,
            successful_samples=successful,
            failed_samples=failed,
            metric_valid_counts=valid_counts,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_key(sample: JudgeSample) -> int:
    if sample.seq is None:
        raise JudgeExecutionError("Judge checkpointing requires every sample to have seq")
    return sample.seq


def _restore_checkpoint(
    checkpoint: dict,
    samples: list[JudgeSample],
    *,
    retry_failed: bool = False,
) -> tuple[list[JudgeDetailedResult], set[int]]:
    raw_detailed = checkpoint.get("detailed")
    if not isinstance(raw_detailed, list):
        raise JudgeExecutionError("Judge checkpoint detailed entries must be a list")

    samples_by_key = {_sample_key(sample): sample for sample in samples}
    restored: list[JudgeDetailedResult] = []
    completed_keys: set[int] = set()
    for raw_entry in raw_detailed:
        try:
            entry = JudgeDetailedResult.model_validate(raw_entry)
        except Exception as exc:
            raise JudgeExecutionError(f"Judge checkpoint entry is invalid: {exc}") from exc
        if entry.seq is None:
            raise JudgeExecutionError("Judge checkpoint entry is missing seq")
        sample = samples_by_key.get(entry.seq)
        if sample is None:
            raise JudgeExecutionError(
                f"Judge checkpoint references unknown prediction row seq={entry.seq}"
            )
        if entry.seq in completed_keys:
            raise JudgeExecutionError(f"Judge checkpoint has duplicate seq={entry.seq}")
        if entry.id != sample.id or entry.question != sample.question:
            raise JudgeExecutionError(
                f"Judge checkpoint does not match prediction row seq={entry.seq}"
            )
        if entry.status == SampleEvaluationStatus.PROGRAMMING_ERROR:
            raise JudgeExecutionError(
                "Judge checkpoint contains a programming error; do not resume it"
            )
        if entry.status == SampleEvaluationStatus.SAMPLE_FAILED:
            if retry_failed:
                continue
            entry.metrics = {}
        validate_metric_mapping(
            entry.metrics,
            scope=f"restored sample id={entry.id} seq={entry.seq}",
        )
        restored.append(entry)
        completed_keys.add(entry.seq)
    return restored, completed_keys


def _compute_averages(scores: dict[str, list[float]]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for metric, values in scores.items():
        valid = [v for v in values if math.isfinite(v)]
        if valid:
            averages[metric] = sum(valid) / len(valid)
    return averages
