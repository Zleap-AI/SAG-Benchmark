"""Strict helpers shared by datasets with supporting-fact references."""

from __future__ import annotations

from typing import Any

from pipeline.evaluation.judge.dataset_adapters.errors import DatasetSchemaError


def _error(dataset: str, row_index: int, field: str, message: str) -> DatasetSchemaError:
    return DatasetSchemaError(f"{dataset} row {row_index}, field {field}: {message}")


def require_question(raw: dict[str, Any], dataset: str, row_index: int) -> str:
    value = raw.get("question")
    if not isinstance(value, str) or not value.strip():
        raise _error(dataset, row_index, "question", "expected a non-empty string")
    return value


def require_answer(raw: dict[str, Any], dataset: str, row_index: int) -> str | list[str]:
    value = raw.get("answer", "")
    if isinstance(value, str) and value.strip():
        base: str | list[str] = value
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        base = value
    else:
        raise _error(
            dataset,
            row_index,
            "answer",
            "expected a non-empty string or non-empty list of non-empty strings",
        )

    # 并入 answer_aliases 作为额外候选；打分时对每个候选取 max，
    # EM 命中任一候选即 1，F1 取最大。
    aliases = raw.get("answer_aliases")
    if isinstance(aliases, list):
        merged = [base] if isinstance(base, str) else list(base)
        added = False
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                merged.append(alias)
                added = True
        if added:
            return merged
    return base


def resolve_supporting_facts(
    raw: dict[str, Any],
    dataset: str,
    row_index: int,
    *,
    sentence_separator: str = "",
) -> tuple[str, ...]:
    """Build title-prefixed gold documents for supporting-fact datasets.

    The benchmark retrieval baselines use ``supporting_facts`` only to select
    context titles. The sentence index remains structurally validated, but it
    deliberately does not choose a sentence: every matching context block is a
    complete gold document. A fact title absent from context produces no
    document, matching the baseline's title-filter behaviour.
    """
    context = raw.get("context")
    facts = raw.get("supporting_facts")
    if not isinstance(context, list):
        raise _error(dataset, row_index, "context", "expected a list")
    if not isinstance(facts, list):
        raise _error(dataset, row_index, "supporting_facts", "expected a list")

    context_blocks: list[tuple[str, list[str]]] = []
    for block_index, block in enumerate(context):
        if not isinstance(block, list) or len(block) != 2:
            raise _error(
                dataset, row_index, f"context[{block_index}]", "expected [title, sentences]"
            )
        title, sentences = block
        if not isinstance(title, str) or not title.strip():
            raise _error(dataset, row_index, f"context[{block_index}][0]", "expected a title")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise _error(
                dataset,
                row_index,
                f"context[{block_index}][1]",
                "expected a list of strings",
            )
        context_blocks.append((title, sentences))

    supporting_titles: set[str] = set()
    for fact_index, fact in enumerate(facts):
        if not isinstance(fact, list) or len(fact) != 2:
            raise _error(
                dataset,
                row_index,
                f"supporting_facts[{fact_index}]",
                "expected [title, sentence_index]",
            )
        title, sentence_index = fact
        if not isinstance(title, str) or not title.strip() or type(sentence_index) is not int:
            raise _error(
                dataset,
                row_index,
                f"supporting_facts[{fact_index}]",
                "expected a string title and integer index",
            )
        supporting_titles.add(title)

    return tuple(
        f"{title}\n{sentence_separator.join(sentences)}"
        for title, sentences in context_blocks
        if title in supporting_titles
    )
