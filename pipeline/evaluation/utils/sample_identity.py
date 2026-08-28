"""Strict identity helpers for external benchmark result files.

External result producers own sample identity. This module deliberately does
not infer an ID from a question, a row position, or a retry counter; callers
must provide the dataset-native identifier explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

RecordT = TypeVar("RecordT", bound=Mapping[str, Any])


def require_sample_id(value: Any, source: str, row_index: int) -> str:
    """Return a non-empty dataset-native sample ID or raise ValueError."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source} row {row_index} must contain a non-empty string sample ID; "
            "do not infer identity from question text or row position"
        )
    return value.strip()


def index_unique_records(
    records: Sequence[RecordT],
    id_field: str,
    source: str,
) -> dict[str, RecordT]:
    """Index records by an explicit ID while rejecting duplicate/invalid rows."""

    indexed: dict[str, RecordT] = {}
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source} row {row_index} must be an object")
        sample_id = require_sample_id(record.get(id_field), source, row_index)
        if sample_id in indexed:
            raise ValueError(
                f"{source} contains duplicate sample ID {sample_id!r} "
                f"at row {row_index}; refusing silent overwrite"
            )
        indexed[sample_id] = record
    return indexed


def validate_question_identity(
    expected: Any,
    actual: Any,
    sample_id: str,
    source: str,
) -> None:
    """Validate the question associated with an already-known sample ID."""

    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"{source} sample {sample_id!r} has an invalid expected question")
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError(f"{source} sample {sample_id!r} has an invalid question")
    if expected.strip() != actual.strip():
        raise ValueError(
            f"{source} sample {sample_id!r} question does not match its source question"
        )


def validate_identity_coverage(
    expected_ids: set[str],
    actual_ids: set[str],
    source: str,
) -> None:
    """Require two identity sets to have exactly the same coverage."""

    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"{source} identity coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
