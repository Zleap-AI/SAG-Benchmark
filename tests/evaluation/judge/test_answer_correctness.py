"""Tests for answer correctness metric with FakeLLM."""

import json

import pytest

from pipeline.evaluation.judge.metrics.answer_correctness import (
    compute_answer_correctness,
    fbeta_score,
)
from pipeline.exceptions import LLMResponseError


class TestFBetaScore:
    def test_perfect(self):
        assert fbeta_score(3, 0, 0) == pytest.approx(1.0, abs=0.01)

    def test_zero(self):
        assert fbeta_score(0, 3, 3) == pytest.approx(0.0, abs=0.01)

    def test_f2_beta(self):
        """F2 weights recall more than precision."""
        f1 = fbeta_score(1, 1, 1, beta=1.0)
        f2 = fbeta_score(1, 1, 1, beta=2.0)
        assert f2 > f1


class TestAnswerCorrectness:
    @pytest.mark.asyncio
    async def test_perfect_match(self):
        from pipeline.core.ai.models import LLMResponse, LLMUsage
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()

        stmt_resp = LLMResponse(
            content='["Paris is the capital of France"]',
            model="fake",
            usage=LLMUsage(),
        )
        class_resp = LLMResponse(
            content=json.dumps(
                {
                    "TP": [
                        {"statement": "Paris is the capital of France", "reason": "Exact match"}
                    ],
                    "FP": [],
                    "FN": [],
                }
            ),
            model="fake",
            usage=LLMUsage(),
        )

        responses = [stmt_resp, stmt_resp, class_resp]
        llm.set_responses([r.content for r in responses])

        score = await compute_answer_correctness(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            ground_truth="Paris is the capital of France.",
            llm=llm,
        )
        # 3 LLM calls: answer stmts, gt stmts, classification
        assert llm.call_count == 3
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_match(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [
                '["Paris is the capital of France"]',
                '["London is the capital of UK"]',
                json.dumps(
                    {
                        "TP": [],
                        "FP": [],
                        "FN": [{"statement": "London is the capital of UK", "reason": "Different"}],
                    }
                ),
            ]
        )

        score = await compute_answer_correctness(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            ground_truth="London is the capital of UK.",
            llm=llm,
        )
        assert score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_with_intermediate(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [
                '["Paris is the capital of France"]',
                '["Paris is the capital of France"]',
                json.dumps(
                    {
                        "TP": [
                            {"statement": "Paris is the capital of France", "reason": "Exact match"}
                        ],
                        "FP": [],
                        "FN": [],
                    }
                ),
            ]
        )

        result = await compute_answer_correctness(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            ground_truth="Paris is the capital of France.",
            llm=llm,
            return_intermediate=True,
        )
        score, intermediate = result
        assert score == pytest.approx(1.0, abs=0.01)
        assert "classification" in intermediate
        assert "answer_statements" in intermediate
        assert "gt_statements" in intermediate

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "classification_response",
        [
            pytest.param("I cannot answer that.", id="not_json"),
            pytest.param("null", id="null"),
            pytest.param('{"TP": "yes", "FP": [], "FN": []}', id="tp_not_a_list"),
            pytest.param('{"TP": [{"statement": "a"}], "FP": [], "FN": []}', id="missing_reason"),
        ],
    )
    async def test_unusable_classification_raises(self, classification_response):
        """An unusable classification must fail the sample rather than score 0.0,
        which is indistinguishable from a genuinely wrong answer. The error type
        matters too: LLMResponseError is an expected failure, so the runner records
        the sample and continues instead of aborting the run."""
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        llm.set_responses(
            [
                '["Paris is the capital of France"]',
                '["Paris is the capital of France"]',
                classification_response,
            ]
        )

        with pytest.raises(LLMResponseError):
            await compute_answer_correctness(
                question="What is the capital of France?",
                answer="Paris is the capital of France.",
                ground_truth="Paris is the capital of France.",
                llm=llm,
            )
