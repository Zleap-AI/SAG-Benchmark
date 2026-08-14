"""Tests for context relevance metric with FakeLLM.

Verifies: two independent LLM calls, averaging, 0-2 scale conversion to 0-1.
"""


import pytest

from pipeline.evaluation.judge.metrics.context_relevance import (
    _is_valid_rating,
    _normalize_rating,
    compute_context_relevance,
)


class TestNormalizeRating:
    def test_dict_rating(self):
        assert _normalize_rating({"rating": 1}) == 1.0
        assert _normalize_rating({"score": 2}) == 2.0

    def test_dict_out_of_range(self):
        assert _normalize_rating({"rating": 3}) is None
        assert _normalize_rating({"rating": -1}) is None

    def test_single_element_list(self):
        assert _normalize_rating([1]) == 1.0

    def test_string_number(self):
        assert _normalize_rating("1") == 1.0

    def test_invalid(self):
        assert _normalize_rating(None) is None
        assert _normalize_rating("not a number") is None
        assert _normalize_rating({"no_rating": 1}) is None


class TestIsValidRating:
    def test_valid(self):
        assert _is_valid_rating(0) is True
        assert _is_valid_rating(1) is True
        assert _is_valid_rating(2) is True
        assert _is_valid_rating(0.5) is True

    def test_invalid(self):
        assert _is_valid_rating(-0.1) is False
        assert _is_valid_rating(2.1) is False
        assert _is_valid_rating("abc") is False
        assert _is_valid_rating(None) is False


class TestContextRelevance:
    @pytest.mark.asyncio
    async def test_two_calls_averaged(self):
        """Verify exactly 2 LLM calls and scores averaged."""
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        # Two independent ratings
        llm.set_responses([
            '{"rating": 2}',
            '{"rating": 0}',
        ])

        score = await compute_context_relevance(
            question="What is the capital of France?",
            contexts=["Paris is in France.", "France is a country."],
            llm=llm,
        )
        # 2 calls, avg of (2/2 + 0/2) = avg(1.0, 0.0) = 0.5
        assert llm.call_count == 2
        assert score == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_question(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        score = await compute_context_relevance(
            question="",
            contexts=["Some context."],
            llm=llm,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_contexts(self):
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        score = await compute_context_relevance(
            question="What is the capital of France?",
            contexts=[],
            llm=llm,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_context_equals_question(self):
        """When context equals question, score should be 0."""
        from tests.evaluation.judge.conftest import FakeLLM

        llm = FakeLLM()
        score = await compute_context_relevance(
            question="What is the capital of France?",
            contexts=["What is the capital of France?"],
            llm=llm,
        )
        assert score == 0.0
