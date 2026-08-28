"""Retrieval Judge evaluation aligned with GraphRAG-Benchmark.

Evaluates context_relevancy and evidence_recall. Evidence is loaded LIVE from
the raw dataset. No embedding/Ragas/Datasets/LangChain dependency.
"""

from typing import Any

from pipeline.core.ai.base import BaseLLMClient
from pipeline.evaluation.judge.metrics import context_relevance, evidence_recall
from pipeline.evaluation.judge.models import (
    EvaluationKind,
    JudgeDetailedResult,
    JudgeSample,
    SampleEvaluationStatus,
)

ALL_RETRIEVAL_METRICS = {"context_relevancy", "evidence_recall"}

# Deterministic ordered tuple for manifests and reproducible results.
ALL_RETRIEVAL_METRICS_SORTED: tuple[str, ...] = (
    "context_relevancy",
    "evidence_recall",
)


def _is_graph_based_context(context: str) -> bool:
    """Return whether context uses the structured graph-section format."""
    return isinstance(context, str) and context.strip().startswith("-----")


def _preprocess_context(context: Any, context_top_k: int = 5) -> str:
    """Preserve graph context and limit plain passage context to top-k chunks."""
    if context is None:
        return ""
    if isinstance(context, list):
        context = "\n\n".join(str(item) for item in context if item)
    if not isinstance(context, str) or not context.strip():
        return ""
    if _is_graph_based_context(context) or context_top_k <= 0:
        return context
    chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    return "\n\n".join(chunks[:context_top_k])


async def evaluate_sample(
    question: str,
    contexts: str | list[str],
    evidences: list[str],
    llm: Any,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval metrics for a single sample."""
    if metrics is None:
        metrics = list(ALL_RETRIEVAL_METRICS)

    results = {}
    if "context_relevancy" in metrics:
        results["context_relevancy"] = await context_relevance.compute_context_relevance(
            question, contexts, llm
        )
    if "evidence_recall" in metrics:
        results["evidence_recall"] = await evidence_recall.compute_evidence_recall(
            question, contexts, evidences, llm
        )

    return results


# ---------------------------------------------------------------------------
# RetrievalSampleEvaluator — implements SampleEvaluator Protocol
# ---------------------------------------------------------------------------


class RetrievalSampleEvaluator:
    """Stateless retrieval evaluator implementing SampleEvaluator Protocol."""

    kind: EvaluationKind = EvaluationKind.RETRIEVAL
    supported_metrics: tuple[str, ...] = ALL_RETRIEVAL_METRICS_SORTED
    default_metrics: tuple[str, ...] = ALL_RETRIEVAL_METRICS_SORTED

    def __init__(
        self,
        evidence_repository: Any = None,
        dataset: str = "",
    ) -> None:
        self._evidence_repo = evidence_repository
        self._dataset = dataset

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
        ev = list(sample.evidences)
        if not ev and self._evidence_repo is not None and self._dataset:
            ev_map = self._evidence_repo.evidence_map(self._dataset)
            ev = list(ev_map.get(sample.id, []))

        result = await evaluate_sample(
            question=sample.question,
            contexts=ctx,
            evidences=ev,
            llm=llm,
            metrics=list(metrics),
        )
        return JudgeDetailedResult(
            id=sample.id,
            seq=sample.seq,
            question=sample.question,
            ground_truth=sample.ground_truth,
            generated_answer=sample.answer,
            contexts=sample.contexts,
            metrics=result,
            llm_intermediate={},
            status=SampleEvaluationStatus.SUCCESS,
        )
