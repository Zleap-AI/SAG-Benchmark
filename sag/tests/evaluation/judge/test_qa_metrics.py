"""Tests for EM/F1 metrics — single and multi-gold, parity with external Judge."""

import pytest

from pipeline.evaluation.judge.generation import (
    _canonical_gt_text,
    _compute_qa_em,
    _compute_qa_f1,
    _preprocess_context,
)
from pipeline.evaluation.utils.eval_utils import normalize_answer


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == normalize_answer("hello world")

    def test_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"

    def test_articles(self):
        assert normalize_answer("The Lord of the Rings") == "lord of rings"

    def test_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_apostrophe_in_word(self):
        norm = normalize_answer("don't")
        assert "dont" in norm


class TestCanonicalGTText:
    def test_plain_str_passthrough(self):
        assert _canonical_gt_text("hello") == "hello"

    def test_empty_str(self):
        assert _canonical_gt_text("") == ""

    def test_whitespace_only_str(self):
        assert _canonical_gt_text("   ") == "   "

    def test_list_first_nonempty(self):
        assert _canonical_gt_text(["", "second", "third"]) == "second"

    def test_list_all_nonempty_returns_first(self):
        assert _canonical_gt_text(["first", "second"]) == "first"

    def test_list_all_empty(self):
        assert _canonical_gt_text(["", "  ", ""]) == ""

    def test_empty_list(self):
        assert _canonical_gt_text([]) == ""

    def test_single_element_list(self):
        assert _canonical_gt_text(["only"]) == "only"

    def test_list_with_whitespace_skipped(self):
        assert _canonical_gt_text(["  ", "\t", "actual"]) == "actual"

    def test_non_str_returns_empty(self):
        assert _canonical_gt_text(42) == ""


class TestContextPreprocessing:
    def test_graph_context_is_preserved(self):
        context = "-----Entities-----\na,b\n\n-----Sources-----\nc,d"
        assert _preprocess_context(context, context_top_k=1) == context

    def test_plain_context_uses_top_k_chunks(self):
        context = "first\n\nsecond\n\nthird"
        assert _preprocess_context(context, context_top_k=2) == "first\n\nsecond"

    def test_zero_disables_plain_context_truncation(self):
        context = "first\n\nsecond"
        assert _preprocess_context(context, context_top_k=0) == context


class TestSingleAnswerEM:
    def test_exact_match(self):
        assert _compute_qa_em("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert _compute_qa_em("Hello World", "hello world") == 1.0

    def test_punctuation_insensitive(self):
        assert _compute_qa_em("Hello, World!", "hello world") == 1.0

    def test_article_insensitive(self):
        assert _compute_qa_em("the hello world", "hello world") == 1.0

    def test_no_match(self):
        assert _compute_qa_em("hello world", "goodbye") == 0.0

    def test_empty_pred(self):
        assert _compute_qa_em("", "hello") == 0.0

    def test_empty_gold(self):
        assert _compute_qa_em("hello", "") == 0.0

    def test_both_empty(self):
        assert _compute_qa_em("", "") == 0.0


class TestSingleAnswerF1:
    def test_perfect_overlap(self):
        assert _compute_qa_f1("cat dog", "cat dog") == 1.0

    def test_partial_overlap(self):
        f1 = _compute_qa_f1("cat dog", "cat bird")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        assert _compute_qa_f1("cat", "dog") == 0.0

    def test_empty_pred(self):
        assert _compute_qa_f1("", "hello") == 0.0

    def test_empty_gold(self):
        assert _compute_qa_f1("hello", "") == 0.0

    def test_repeated_tokens(self):
        f1 = _compute_qa_f1("cat cat", "cat")
        assert f1 == pytest.approx(2 / 3, abs=0.01)


class TestMultiAnswerEM:
    def test_any_hit(self):
        assert _compute_qa_em("paris", ["london", "paris", "berlin"]) == 1.0

    def test_all_miss(self):
        assert _compute_qa_em("rome", ["london", "paris", "berlin"]) == 0.0

    def test_empty_list(self):
        assert _compute_qa_em("paris", []) == 0.0

    def test_with_empty_strings(self):
        assert _compute_qa_em("paris", ["", "paris", ""]) == 1.0

    def test_all_empty_strings(self):
        assert _compute_qa_em("paris", ["", ""]) == 0.0


class TestMultiAnswerF1:
    def test_max_across_golds(self):
        f1_best = _compute_qa_f1("cat dog bird", ["cat dog", "cat dog bird fish"])
        f1_single = _compute_qa_f1("cat dog bird", "cat dog bird fish")
        assert f1_best == f1_single

    def test_empty_list(self):
        assert _compute_qa_f1("paris", []) == 0.0

    def test_with_empty_strings(self):
        f1 = _compute_qa_f1("paris", ["", "paris"])
        assert f1 == 1.0
