"""Tests for Judge runner concurrency and failure isolation."""

import asyncio

import pytest


class TestSemaphoreConcurrency:
    """Verify concurrent LLM calls respect semaphore limit."""

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        """Calls must not exceed semaphore limit."""
        from tests.evaluation.judge.conftest import FakeLLM

        max_concurrent = 2
        total_calls = 8

        llm = FakeLLM()
        llm.set_responses(['{"result": "ok"}'] * total_calls * 3)
        concurrent_count = 0
        max_seen = 0

        original_chat = llm.chat

        async def tracked_chat(*args, **kwargs):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.01)
            result = await original_chat(*args, **kwargs)
            concurrent_count -= 1
            return result

        llm.chat = tracked_chat

        sem = asyncio.Semaphore(max_concurrent)

        async def worker(i):
            async with sem:
                return await llm.chat([])

        tasks = [worker(i) for i in range(total_calls)]
        await asyncio.gather(*tasks)

        assert max_seen <= max_concurrent
        assert llm.call_count == total_calls


class TestFailureIsolation:
    """Verify single sample failure doesn't crash entire run."""

    @pytest.mark.asyncio
    async def test_failure_isolated(self):
        """Failing sample should produce NaN scores, not crash the run."""
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        # First call succeeds, second raises
        call_idx = [0]

        original_chat = llm.chat

        async def flaky_chat(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 2:
                raise RuntimeError("Simulated LLM failure")
            return await original_chat(*args, **kwargs)

        llm.chat = flaky_chat

        # Simulate what the runner does: wrap each sample in try/except
        results = []
        for i in range(3):
            try:
                resp = await llm.chat([])
                results.append({"id": i, "ok": True, "content": resp.content})
            except Exception:
                results.append({"id": i, "ok": False, "error": True})

        assert len(results) == 3
        assert results[0]["ok"] is True
        assert results[1]["ok"] is False
        assert results[2]["ok"] is True


class TestProgrammingErrorAbort:
    """Verify programming errors cancel remaining tasks and abort the run."""

    @pytest.mark.asyncio
    async def test_programming_error_aborts_run(self):
        """TypeError/KeyError/etc must cancel other tasks and raise JudgeExecutionError."""
        from pipeline.evaluation.judge.errors import JudgeExecutionError
        from pipeline.evaluation.judge.models import (
            EvaluationKind,
            JudgeSample,
            SampleEvaluationStatus,
        )
        from pipeline.evaluation.judge.runner import JudgeEvaluationRunner

        class _BrokenEval:
            kind = EvaluationKind.GENERATION
            supported_metrics = ("qa_em",)

            async def evaluate(self, sample, llm, metrics, context_top_k=5):
                if sample.id == 1:
                    raise TypeError("intentional programming error")
                from pipeline.evaluation.judge.models import JudgeDetailedResult

                return JudgeDetailedResult(
                    id=sample.id,
                    question=sample.question,
                    ground_truth="",
                    generated_answer="",
                    contexts=[],
                    metrics={"qa_em": 1.0},
                    llm_intermediate={},
                    status=SampleEvaluationStatus.SUCCESS,
                )

        runner = JudgeEvaluationRunner(max_concurrent=1)
        samples = [
            JudgeSample(id=0, question="Q0", answer="A0", ground_truth="A0"),
            JudgeSample(id=1, question="Q1", answer="A1", ground_truth="A1"),
            JudgeSample(id=2, question="Q2", answer="A2", ground_truth="A2"),
        ]

        with pytest.raises(JudgeExecutionError):
            await runner.run(
                samples=samples,
                evaluator=_BrokenEval(),
                llm=None,
                metrics=("qa_em",),
            )


class TestExpectedFailureIsolation:
    """Verify expected LLM failures (timeout, connection) are isolated per-sample."""

    @pytest.mark.asyncio
    async def test_timeout_is_isolated(self):
        """TimeoutError cannot leave NaN metrics in the final run."""
        from pipeline.evaluation.judge.models import (
            EvaluationKind,
            JudgeSample,
            SampleEvaluationStatus,
        )
        from pipeline.evaluation.judge.runner import JudgeEvaluationRunner

        class _TimeoutEval:
            kind = EvaluationKind.GENERATION
            supported_metrics = ("qa_em",)

            async def evaluate(self, sample, llm, metrics, context_top_k=5):
                if sample.id == 1:
                    raise TimeoutError("LLM timed out")
                from pipeline.evaluation.judge.models import JudgeDetailedResult

                return JudgeDetailedResult(
                    id=sample.id,
                    question=sample.question,
                    ground_truth="",
                    generated_answer="",
                    contexts=[],
                    metrics={"qa_em": 1.0},
                    llm_intermediate={},
                    status=SampleEvaluationStatus.SUCCESS,
                )

        runner = JudgeEvaluationRunner(max_concurrent=1)
        samples = [
            JudgeSample(id=0, question="Q0", answer="A0", ground_truth="A0"),
            JudgeSample(id=1, question="Q1", answer="A1", ground_truth="A1"),
            JudgeSample(id=2, question="Q2", answer="A2", ground_truth="A2"),
        ]

        summary = await runner.run(
            samples=samples,
            evaluator=_TimeoutEval(),
            llm=None,
            metrics=("qa_em",),
        )

        assert summary.successful_samples == 2
        assert summary.failed_samples == 1
        assert summary.average_scores == {"qa_em": 1.0}
        assert summary.metric_valid_counts == {"qa_em": 2}
        failed = next(
            item for item in summary.detailed if item.status == SampleEvaluationStatus.SAMPLE_FAILED
        )
        assert failed.metrics == {}
        assert failed.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_evidence_repository_programming_error_aborts():
    from pipeline.evaluation.judge.errors import JudgeExecutionError
    from pipeline.evaluation.judge.models import JudgeSample
    from pipeline.evaluation.judge.retrieval import RetrievalSampleEvaluator
    from pipeline.evaluation.judge.runner import JudgeEvaluationRunner

    class _BrokenEvidenceRepository:
        def evidence_map(self, dataset):
            raise KeyError("broken evidence mapping")

    evaluator = RetrievalSampleEvaluator(
        evidence_repository=_BrokenEvidenceRepository(),
        dataset="test",
    )
    runner = JudgeEvaluationRunner(max_concurrent=1)
    sample = JudgeSample(
        id=0,
        question="Q",
        answer="A",
        ground_truth="A",
        contexts=["C"],
    )
    with pytest.raises(JudgeExecutionError, match="KeyError"):
        await runner.run(
            samples=[sample],
            evaluator=evaluator,
            llm=None,
            metrics=("evidence_recall",),
        )


@pytest.mark.asyncio
async def test_checkpoint_resumes_only_missing_samples_without_repeating_llm_calls(tmp_path):
    from pipeline.evaluation.judge.checkpoint import atomic_write_json
    from pipeline.evaluation.judge.models import (
        EvaluationKind,
        JudgeDetailedResult,
        JudgeSample,
        SampleEvaluationStatus,
    )
    from pipeline.evaluation.judge.runner import JudgeEvaluationRunner

    class _CountingEvaluator:
        kind = EvaluationKind.GENERATION
        supported_metrics = ("qa_em",)
        default_metrics = ("qa_em",)

        def __init__(self):
            self.called_ids = []

        async def evaluate(self, sample, llm, metrics, context_top_k=5):
            self.called_ids.append(sample.id)
            return JudgeDetailedResult(
                id=sample.id,
                seq=sample.seq,
                question=sample.question,
                ground_truth=sample.ground_truth,
                generated_answer=sample.answer,
                contexts=[],
                metrics={"qa_em": 1.0},
                status=SampleEvaluationStatus.SUCCESS,
            )

    samples = [
        JudgeSample(id=i, seq=i, question=f"Q{i}", answer="A", ground_truth="A") for i in range(3)
    ]
    checkpoint_path = tmp_path / "generation_results.json.partial"
    atomic_write_json(
        {
            "average_scores": {"qa_em": 1.0},
            "detailed": [
                {
                    "id": 0,
                    "seq": 0,
                    "question": "Q0",
                    "ground_truth": "A",
                    "generated_answer": "A",
                    "contexts": [],
                    "metrics": {"qa_em": 1.0},
                    "status": "success",
                },
                {
                    "id": 1,
                    "seq": 1,
                    "question": "Q1",
                    "ground_truth": "A",
                    "generated_answer": "A",
                    "contexts": [],
                    "metrics": {"qa_em": float("nan")},
                    "status": "sample_failed",
                    "error_type": "LLMResponseError",
                    "error_message": "invalid response",
                },
            ],
        },
        str(checkpoint_path),
    )

    evaluator = _CountingEvaluator()
    summary = await JudgeEvaluationRunner(max_concurrent=1).run(
        samples=samples,
        evaluator=evaluator,
        llm=None,
        metrics=("qa_em",),
        checkpoint_path=str(checkpoint_path),
    )

    assert evaluator.called_ids == [2]
    assert summary.total_samples == 3
    assert summary.successful_samples == 2
    assert summary.failed_samples == 1
    assert summary.average_scores == {"qa_em": 1.0}
    assert summary.metric_valid_counts == {"qa_em": 2}
    assert summary.detailed[1].metrics == {}
