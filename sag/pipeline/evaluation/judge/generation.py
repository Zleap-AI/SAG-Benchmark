"""Generation Judge evaluation aligned with GraphRAG-Benchmark.

Evaluates generation quality metrics: EM, F1, ROUGE, answer_correctness,
and coverage_score. No embedding/datasets/LangChain dependency.
"""

import asyncio
from typing import Any

from pipeline.core.ai.base import BaseLLMClient
from pipeline.evaluation.judge.metrics import (
    answer_correctness,
    coverage,
    rouge,
)
from pipeline.evaluation.judge.models import (
    EvaluationKind,
    JudgeDetailedResult,
    JudgeSample,
    SampleEvaluationStatus,
)
from pipeline.evaluation.utils.eval_utils import normalize_answer

ALL_GENERATION_METRICS = {
    "qa_em",
    "qa_f1",
    "rouge_score",
    "answer_correctness",
    "coverage_score",
}

# Deterministic ordered tuple for manifests and reproducible results.
ALL_GENERATION_METRICS_SORTED: tuple[str, ...] = (
    "answer_correctness",
    "coverage_score",
    "qa_em",
    "qa_f1",
    "rouge_score",
)

# User-approved five-metric default QA set.
DEFAULT_GENERATION_METRICS: tuple[str, ...] = (
    "answer_correctness",
    "coverage_score",
    "qa_em",
    "qa_f1",
    "rouge_score",
)

# Question-type routing from the reference, limited to supported metrics.
GENERATION_METRIC_CONFIG: dict[str, tuple[str, ...]] = {
    "Fact Retrieval": ("rouge_score", "answer_correctness", "qa_f1", "qa_em"),
    "Complex Reasoning": ("rouge_score", "answer_correctness", "qa_f1", "qa_em"),
    "Contextual Summarize": ("answer_correctness", "coverage_score", "qa_f1", "qa_em"),
    "Creative Generation": (
        "answer_correctness",
        "coverage_score",
        "qa_f1",
        "qa_em",
    ),
    "qa": (
        "rouge_score",
        "answer_correctness",
        "coverage_score",
        "qa_f1",
        "qa_em",
    ),
}


def metrics_for_question_type(
    question_type: str,
    requested_metrics: tuple[str, ...],
) -> tuple[str, ...]:
    """Apply the authoritative per-question-type metric routing."""
    allowed = GENERATION_METRIC_CONFIG.get(question_type, ())
    return tuple(metric for metric in allowed if metric in requested_metrics)


def _is_graph_based_context(context: str) -> bool:
    """Return whether context uses the structured graph-section format."""
    return isinstance(context, str) and context.strip().startswith("-----")


def _preprocess_context(
    context: str | list[str],
    context_top_k: int = 5,
) -> str:
    """Preserve graph context and limit plain passage context to top-k chunks."""
    if isinstance(context, list):
        context = "\n\n".join(str(item) for item in context if item)
    if not isinstance(context, str) or not context.strip():
        return ""
    if _is_graph_based_context(context) or context_top_k <= 0:
        return context
    chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    return "\n\n".join(chunks[:context_top_k])


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
    contexts: str | list[str],
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


# ---------------------------------------------------------------------------
# GenerationSampleEvaluator — implements SampleEvaluator Protocol
# ---------------------------------------------------------------------------


class GenerationSampleEvaluator:
    """Stateless generation evaluator implementing SampleEvaluator Protocol."""

    kind: EvaluationKind = EvaluationKind.GENERATION
    supported_metrics: tuple[str, ...] = ALL_GENERATION_METRICS_SORTED
    default_metrics: tuple[str, ...] = DEFAULT_GENERATION_METRICS

    async def evaluate(
        self,
        sample: JudgeSample,
        llm: BaseLLMClient | None,
        metrics: tuple[str, ...],
        context_top_k: int = 5,
    ) -> JudgeDetailedResult:
        ctx = _preprocess_context(
            sample.contexts if sample.contexts else [],
            context_top_k,
        )
        sample_metrics = metrics_for_question_type(sample.question_type, metrics)
        result = await evaluate_sample(
            question=sample.question,
            answer=sample.answer,
            contexts=ctx,
            ground_truth=sample.ground_truth,
            metrics=list(sample_metrics),
            llm=llm,
            detailed_output=True,
        )
        return JudgeDetailedResult(
            id=sample.id,
            seq=sample.seq,
            question=sample.question,
            ground_truth=sample.ground_truth,
            generated_answer=sample.answer,
            contexts=sample.contexts,
            metrics=result["scores"],
            llm_intermediate=result.get("llm_intermediate"),
            status=SampleEvaluationStatus.SUCCESS,
        )
