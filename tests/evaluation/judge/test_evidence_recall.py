"""Tests for evidence recall metric with FakeLLM."""

import json

import numpy as np
import pytest

from pipeline.evaluation.judge.metrics.evidence_recall import compute_evidence_recall


class TestEvidenceRecall:
    @pytest.mark.asyncio
    async def test_full_recall(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({
                "classifications": [
                    {"statement": "Paris is the capital of France.", "reason": "Found", "attributed": 1},
                    {"statement": "France is in Europe.", "reason": "Found", "attributed": 1},
                ]
            }),
        ])

        score = await compute_evidence_recall(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France. France is in Europe."],
            reference_evidence=[
                "Paris is the capital of France.",
                "France is in Europe.",
            ],
            llm=llm,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_partial_recall(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({
                "classifications": [
                    {"statement": "Paris is the capital of France.", "reason": "Found", "attributed": 1},
                    {"statement": "Paris has the Eiffel Tower.", "reason": "Not found", "attributed": 0},
                ]
            }),
        ])

        score = await compute_evidence_recall(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            reference_evidence=[
                "Paris is the capital of France.",
                "Paris has the Eiffel Tower.",
            ],
            llm=llm,
        )
        assert score == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_evidence(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses([
            json.dumps({"classifications": []}),
        ])

        score = await compute_evidence_recall(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            reference_evidence=[],
            llm=llm,
        )
        # Empty evidence list means no classifications -> NaN
        assert np.isnan(score)

    @pytest.mark.asyncio
    async def test_empty_context(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        score = await compute_evidence_recall(
            question="What is the capital of France?",
            contexts=[],
            reference_evidence=["Paris is the capital of France."],
            llm=llm,
        )
        assert score == 0.0
