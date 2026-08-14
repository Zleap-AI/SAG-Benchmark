"""Coverage Score — migrated from external/judge/Evaluation/metrics/coverage.py.

Measures what percentage of reference facts are covered in the response.
"""

import json
from typing import Any

import numpy as np

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.parser import parse_with_fallbacks


async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm: Any,
    max_retries: int = 2,
    return_intermediate: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    if not reference.strip():
        intermediate = {"facts": [], "classifications": []}
        return (1.0, intermediate) if return_intermediate else 1.0

    facts = await _extract_facts(question, reference, llm, max_retries)

    if not facts:
        intermediate = {"facts": [], "classifications": []}
        return (np.nan, intermediate) if return_intermediate else np.nan

    coverage = await _check_fact_coverage(
        question, facts, response, llm, max_retries
    )

    if coverage:
        attributed = [c["attributed"] for c in coverage]
        score = sum(attributed) / len(attributed)
    else:
        score = np.nan

    intermediate = {"facts": facts, "classifications": coverage}
    return (score, intermediate) if return_intermediate else score


async def _extract_facts(
    question: str,
    reference: str,
    llm: Any,
    max_retries: int,
) -> list[str]:
    prompt = prompts.FACT_EXTRACTION_PROMPT.format(
        question=question, reference=reference[:3000]
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.chat(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.0
            )
            parsed = await parse_with_fallbacks(response.content)
            return _validate_facts(
                parsed.get("facts", []) if isinstance(parsed, dict) else parsed
            )
        except Exception:
            continue
    return []


def _validate_facts(facts: list) -> list[str]:
    if not isinstance(facts, list):
        return []
    return [
        str(f).strip()
        for f in facts
        if isinstance(f, (str, int, float)) and str(f).strip()
    ]


async def _check_fact_coverage(
    question: str,
    facts: list[str],
    response_text: str,
    llm: Any,
    max_retries: int,
) -> list[dict[str, Any]]:
    prompt = prompts.FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response_text[:3000],
        facts=json.dumps(facts),
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.chat(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.0
            )
            parsed = await parse_with_fallbacks(response.content)
            return _validate_classifications(
                parsed.get("classifications", [])
                if isinstance(parsed, dict)
                else parsed
            )
        except Exception:
            continue
    return []


def _validate_classifications(classifications: list) -> list[dict[str, Any]]:
    if not isinstance(classifications, list):
        return []
    valid = []
    for item in classifications:
        if not isinstance(item, dict):
            continue
        try:
            if (
                "statement" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "attributed": int(item["attributed"]),
                })
        except (TypeError, ValueError):
            continue
    return valid
