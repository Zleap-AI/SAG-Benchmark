"""Tests for the final metric-value integrity boundary."""

import math

import pytest

from pipeline.evaluation.judge.errors import MetricResultError
from pipeline.evaluation.judge.metric_validation import validate_metric_mapping


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True, None])
def test_non_finite_or_non_numeric_metric_is_rejected(value, caplog):
    with caplog.at_level("ERROR"):
        with pytest.raises(MetricResultError, match="Invalid metric result"):
            validate_metric_mapping({"qa_f1": value}, scope="sample id=7")

    assert "sample id=7" in caplog.text
    assert "qa_f1" in caplog.text


def test_finite_integer_and_float_metrics_are_accepted():
    validate_metric_mapping({"qa_em": 1, "qa_f1": 0.75}, scope="sample id=1")


def test_nan_check_is_not_python_truthiness_based():
    assert math.isnan(float("nan"))
    with pytest.raises(MetricResultError):
        validate_metric_mapping({"qa_em": float("nan")}, scope="run average_scores")
