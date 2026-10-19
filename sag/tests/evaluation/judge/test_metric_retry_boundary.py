"""Judge metrics delegate request retries to the shared LLM client."""

import pytest

from pipeline.evaluation.judge.metrics.evidence_recall import compute_evidence_recall
from pipeline.exceptions import LLMRequestError, LLMResponseError


class _FailingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _messages, **_kwargs):
        self.calls += 1
        raise LLMRequestError("bad request")


@pytest.mark.asyncio
async def test_metric_propagates_request_failure_without_local_retry(monkeypatch):
    # No chunking/retry loop: a request failure propagates directly, one call only.
    llm = _FailingLLM()

    with pytest.raises(LLMRequestError, match="bad request"):
        await compute_evidence_recall(
            question="Q",
            contexts=["context"],
            reference_evidence=["evidence"],
            llm=llm,
        )

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_runner_records_response_contract_failure(caplog):
    from pipeline.evaluation.judge.models import (
        EvaluationKind,
        JudgeSample,
        SampleEvaluationStatus,
    )
    from pipeline.evaluation.judge.runner import JudgeEvaluationRunner

    class _InvalidResponseEvaluator:
        kind = EvaluationKind.RETRIEVAL
        supported_metrics = ("evidence_recall",)
        default_metrics = ("evidence_recall",)

        async def evaluate(self, sample, llm, metrics, context_top_k=5):
            raise LLMResponseError("invalid metric response")

    runner = JudgeEvaluationRunner(max_concurrent=1)
    sample = JudgeSample(id=0, question="Q", answer="A", ground_truth="A")

    summary = await runner.run(
        samples=[sample],
        evaluator=_InvalidResponseEvaluator(),
        llm=None,
        metrics=("evidence_recall",),
    )

    assert summary.successful_samples == 0
    assert summary.failed_samples == 1
    assert summary.average_scores == {}
    assert summary.metric_valid_counts == {"evidence_recall": 0}
    assert summary.detailed[0].status == SampleEvaluationStatus.SAMPLE_FAILED
    assert summary.detailed[0].metrics == {}
    assert any("failed with expected LLMResponseError" in message for message in caplog.messages)
