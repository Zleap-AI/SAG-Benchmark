"""Tests for coverage metric with FakeLLM."""

import json

import pytest

from pipeline.evaluation.judge.metrics.coverage import compute_coverage_score


class TestCoverageScore:
    @pytest.mark.asyncio
    async def test_full_coverage(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({"facts": ["Paris is the capital of France", "France is in Europe"]}),
            json.dumps({
                "classifications": [
                    {"statement": "Paris is the capital of France", "attributed": 1},
                    {"statement": "France is in Europe", "attributed": 1},
                ]
            }),
        ])

        score = await compute_coverage_score(
            question="What is the capital of France?",
            reference="Paris is the capital of France. France is in Europe.",
            response="Paris is the capital of France and France is in Europe.",
            llm=llm,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_partial_coverage(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({"facts": ["Paris is the capital of France", "France is in Europe"]}),
            json.dumps({
                "classifications": [
                    {"statement": "Paris is the capital of France", "attributed": 1},
                    {"statement": "France is in Europe", "attributed": 0},
                ]
            }),
        ])

        score = await compute_coverage_score(
            question="What is the capital of France?",
            reference="Paris is the capital of France. France is in Europe.",
            response="Paris is the capital of France.",
            llm=llm,
        )
        assert score == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        score = await compute_coverage_score(
            question="What is the capital of France?",
            reference="",
            response="Some answer.",
            llm=llm,
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_with_intermediate(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({"facts": ["Paris is the capital of France"]}),
            json.dumps({
                "classifications": [
                    {"statement": "Paris is the capital of France", "attributed": 1},
                ]
            }),
        ])

        result = await compute_coverage_score(
            question="What is the capital of France?",
            reference="Paris is the capital of France.",
            response="Paris is the capital of France.",
            llm=llm,
            return_intermediate=True,
        )
        score, intermediate = result
        assert score == pytest.approx(1.0, abs=0.01)
        assert "facts" in intermediate
        assert "classifications" in intermediate
