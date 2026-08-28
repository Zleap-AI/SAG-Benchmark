"""2Wiki evidence construction aligned with benchmark title-level recall."""

from __future__ import annotations

from typing import Any

from pipeline.evaluation.judge.dataset_adapters.common import resolve_supporting_facts


def resolve_two_wiki_supporting_facts(
    raw: dict[str, Any],
    dataset: str,
    row_index: int,
) -> tuple[str, ...]:
    """Return complete title-prefixed context documents for 2Wiki facts.

    The sentence index is type-checked by the shared resolver but intentionally
    ignored for selection, so released out-of-range references retain their
    matching full context document.
    """
    return resolve_supporting_facts(
        raw,
        dataset,
        row_index,
        sentence_separator=" ",
    )
