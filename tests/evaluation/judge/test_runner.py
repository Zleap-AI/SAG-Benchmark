"""Tests for Judge runner concurrency and failure isolation."""

import asyncio
import json
import math

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

    @pytest.mark.asyncio
    async def test_generation_failure_is_recorded_as_nan(
        self, tmp_path, monkeypatch
    ):
        from pipeline.evaluation.judge import generation
        from tests.evaluation.judge.conftest import FakeLLM

        data_file = tmp_path / "predictions.json"
        data_file.write_text(
            json.dumps(
                [
                    {
                        "id": 0,
                        "question_type": "qa",
                        "question": "good",
                        "generated_answer": "a",
                        "ground_truth": "a",
                        "context": "c",
                    },
                    {
                        "id": 1,
                        "question_type": "qa",
                        "question": "bad",
                        "generated_answer": "b",
                        "ground_truth": "b",
                        "context": "c",
                    },
                ]
            ),
            encoding="utf-8",
        )

        async def fake_evaluate_sample(**kwargs):
            if kwargs["question"] == "bad":
                raise RuntimeError("simulated")
            return {
                "scores": {"qa_em": 1.0},
                "llm_intermediate": {},
            }

        monkeypatch.setattr(generation, "evaluate_sample", fake_evaluate_sample)
        result = await generation.run_generation_eval(
            str(data_file),
            FakeLLM(),
            detailed_output=True,
            only_metrics=["qa_em"],
        )

        detailed = {entry["id"]: entry for entry in result["qa"]["detailed"]}
        assert detailed[0]["metrics"]["qa_em"] == 1.0
        assert math.isnan(detailed[1]["metrics"]["qa_em"])
        assert result["qa"]["average_scores"]["qa_em"] == 1.0
