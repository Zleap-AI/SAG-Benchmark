"""Context Relevance — migrated from external/judge/Evaluation/metrics/context_relevance.py.

Each context is independently rated twice (0-2 scale), converted to 0-1, then averaged.
"""

import json
from typing import Any

import numpy as np

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.parser import parse_with_fallbacks


async def compute_context_relevance(
    question: str,
    contexts: list[str],
    llm: Any,
    max_retries: int = 2,
) -> float:
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0

    if isinstance(contexts, list):
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0

    prompt = prompts.CONTEXT_RELEVANCE_PROMPT.format(
        question=question, context=context_str
    )

    rating1 = await _get_llm_rating(prompt, llm, max_retries)
    rating2 = await _get_llm_rating(prompt, llm, max_retries)

    scores = [r / 2 for r in [rating1, rating2] if r is not None]
    if not scores:
        return np.nan
    return sum(scores) / len(scores)


async def _get_llm_rating(
    prompt: str,
    llm: Any,
    max_retries: int,
) -> float | None:
    for _ in range(max_retries):
        try:
            response = await llm.chat(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.0
            )
            parsed = await parse_with_fallbacks(response.content)
            return _normalize_rating(parsed)
        except Exception:
            continue
    return None


def _normalize_rating(parsed: dict | list | str | None) -> float | None:
    if isinstance(parsed, dict):
        score = parsed.get("rating", parsed.get("score"))
        if _is_valid_rating(score):
            return float(score)

    if isinstance(parsed, list) and len(parsed) == 1:
        if _is_valid_rating(parsed[0]):
            return float(parsed[0])

    if isinstance(parsed, str):
        stripped = parsed.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                score = data.get("rating", data.get("score"))
                if _is_valid_rating(score):
                    return float(score)
        except Exception:
            pass
        try:
            value = float(stripped)
            if _is_valid_rating(value):
                return value
        except ValueError:
            pass
        for token in stripped.split()[:8]:
            try:
                value = float(token)
                if _is_valid_rating(value):
                    return value
            except ValueError:
                continue
    return None


def _is_valid_rating(value) -> bool:
    try:
        ivalue = float(value)
        return 0 <= ivalue <= 2
    except (TypeError, ValueError):
        return False
