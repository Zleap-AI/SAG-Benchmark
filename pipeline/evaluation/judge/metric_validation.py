"""Strict validation for metric values at Judge execution boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from pipeline.evaluation.judge.errors import MetricResultError
from pipeline.utils import get_logger

logger = get_logger(__name__)


def validate_metric_mapping(
    metrics: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    """Reject non-numeric, NaN, and infinite metric values.

    Metric functions may fail internally and return ``numpy.nan``. Allowing
    that value into JSON or an average makes a run look complete while hiding
    the failed metric, so this boundary logs the exact location and aborts.
    """
    if not isinstance(metrics, Mapping):
        _raise_invalid(scope, "<mapping>", metrics)
    for metric, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            _raise_invalid(scope, metric, value)
        numeric = float(value)
        if not math.isfinite(numeric):
            _raise_invalid(scope, metric, value)


def _raise_invalid(scope: str, metric: str, value: Any) -> None:
    message = (
        f"Invalid metric result: scope={scope}, metric={metric!r}, "
        f"value={value!r}; expected a finite number"
    )
    logger.error(message)
    raise MetricResultError(message)
