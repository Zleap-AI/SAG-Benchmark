"""Tests for ROUGE score computation."""

import pytest

from pipeline.evaluation.judge.metrics.rouge import compute_rouge_score


class TestRougeScore:
    @pytest.mark.asyncio
    async def test_perfect_match(self):
        score = await compute_rouge_score("hello world", "hello world")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        score = await compute_rouge_score("hello world", "goodbye universe")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        score = await compute_rouge_score("hello world today", "hello world")
        assert 0.0 < score < 1.0

    @pytest.mark.asyncio
    async def test_empty_answer(self):
        score = await compute_rouge_score("", "hello world")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_ground_truth(self):
        score = await compute_rouge_score("hello world", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_both_empty(self):
        score = await compute_rouge_score("", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_rouge1_type(self):
        score = await compute_rouge_score("the cat sat", "the cat sat", rouge_type="rouge1")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_precision_mode(self):
        score = await compute_rouge_score("the cat sat on the mat", "the cat sat", mode="precision")
        assert 0.0 < score <= 1.0

    @pytest.mark.asyncio
    async def test_recall_mode(self):
        score = await compute_rouge_score("the cat", "the cat sat on the mat", mode="recall")
        assert 0.0 < score <= 1.0
