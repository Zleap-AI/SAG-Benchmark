"""Context Relevance aligned with GraphRAG-Benchmark/Evaluation/metrics/context_relevance.py.

Each context is independently rated twice (0-2 scale), converted to 0-1, then averaged.
No context chunking — the full context is passed to the LLM in a single call, matching
the authoritative implementation.

Unlike the reference, a sample whose ratings all fail raises instead of yielding NaN,
so the sample is recorded as failed rather than aborting the run.
"""

import json
from typing import Any

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.parser import parse_with_fallbacks
from pipeline.exceptions import LLMResponseError


async def compute_context_relevance(
    question: str,
    contexts: str | list[str],
    llm: Any,
) -> float:
    if isinstance(contexts, list):
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    if not question.strip() or not context_str.strip():
        return 0.0

    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0

    prompt = prompts.CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        context=context_str,
    )

    rating1 = await _get_llm_rating(prompt, llm)
    rating2 = await _get_llm_rating(prompt, llm)

    scores = [r / 2 for r in [rating1, rating2] if r is not None]
    if not scores:
        raise LLMResponseError("Context relevance returned no valid ratings")
    return sum(scores) / len(scores)


async def _get_llm_rating(prompt: str, llm: Any) -> float | None:
    try:
        response = await llm.chat(
            [LLMMessage(role=LLMRole.USER, content=prompt)],
            temperature=0.0,
            seed=42,
        )
        parsed = await parse_with_fallbacks(response.content)
        return _normalize_rating(parsed)
    except Exception:
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


def _is_valid_rating(value: object) -> bool:
    try:
        ivalue = float(value)  # type: ignore[arg-type]
        return 0 <= ivalue <= 2
    except (TypeError, ValueError):
        return False
