"""Tests for evidence recall metric with FakeLLM.

Matches the authoritative evidence_recall implementation: `statement` field,
loose validation (invalid items dropped, not raised), no context chunking,
and `sum(attributed) / len(valid)` denominator.
"""

import json

import pytest

from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.errors import MetricResultError
from pipeline.evaluation.judge.metrics.evidence_recall import compute_evidence_recall
from pipeline.exceptions import LLMResponseError


class TestEvidenceRecall:
    @pytest.mark.asyncio
    async def test_full_recall(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [
                json.dumps(
                    {
                        "classifications": [
                            {
                                "statement": "Paris is the capital of France.",
                                "reason": "Found",
                                "attributed": 1,
                            },
                            {
                                "statement": "France is in Europe.",
                                "reason": "Found",
                                "attributed": 1,
                            },
                        ]
                    }
                ),
            ]
        )

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
        llm.set_responses(
            [
                json.dumps(
                    {
                        "classifications": [
                            {
                                "statement": "Paris has the Eiffel Tower.",
                                "reason": "Not found",
                                "attributed": 0,
                            },
                            {
                                "statement": "Paris is the capital of France.",
                                "reason": "Found",
                                "attributed": 1,
                            },
                        ]
                    }
                ),
            ]
        )

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
        llm.set_responses(
            [
                json.dumps({"classifications": []}),
            ]
        )

        with pytest.raises(MetricResultError, match="at least one canonical evidence"):
            await compute_evidence_recall(
                question="What is the capital of France?",
                contexts=["Paris is the capital of France."],
                reference_evidence=[],
                llm=llm,
            )

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

    @pytest.mark.asyncio
    async def test_invalid_classification_is_dropped(self):
        """Loose validation: an item missing `statement` is dropped, not raised."""
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [
                json.dumps(
                    {
                        "classifications": [
                            {
                                "statement": "Paris is the capital of France.",
                                "reason": "Found",
                                "attributed": 1,
                            },
                            {
                                # missing statement -> dropped
                                "reason": "orphan",
                                "attributed": 1,
                            },
                        ]
                    }
                )
            ]
        )

        score = await compute_evidence_recall(
            question="Q",
            contexts=["Document A"],
            reference_evidence=["Paris is the capital of France.", "Paris has the Eiffel Tower."],
            llm=llm,
        )
        # denominator = 1 valid classification -> 1/1
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_valid_classifications_raises(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [json.dumps({"classifications": [{"reason": "no statement", "attributed": 1}]})]
        )

        with pytest.raises(LLMResponseError, match="no valid classifications"):
            await compute_evidence_recall(
                question="Q",
                contexts=["Document A"],
                reference_evidence=["Document A"],
                llm=llm,
            )

    def test_evidence_recall_context_is_a_quoted_string(self):
        rendered = prompts.EVIDENCE_RECALL_PROMPT.format(
            question="Q",
            context='A long context with an embedded "quote".',
            evidence='["Document A"]',
        )

        # Context is wrapped in quotes (authoritative format), not a bare block.
        assert 'Context: "A long context' in rendered
        assert "Context:\nA long context" not in rendered

    def test_evidence_recall_uses_statement_field_not_evidence_id(self):
        rendered = prompts.EVIDENCE_RECALL_PROMPT.format(
            question="Q",
            context="ctx",
            evidence='["Document A"]',
        )

        assert '"statement"' in rendered
        assert "evidence_id" not in rendered
