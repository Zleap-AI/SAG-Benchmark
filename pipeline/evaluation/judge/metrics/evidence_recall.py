"""Evidence Recall — migrated from external/judge/Evaluation/metrics/evidence_recall.py.

Measures what percentage of reference evidence is supported by the context.
Evidence is loaded LIVE from the raw dataset, not from predictions.
"""

from typing import Any

import numpy as np

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.parser import parse_with_fallbacks


async def compute_evidence_recall(
    question: str,
    contexts: list[str],
    reference_evidence: list[str],
    llm: Any,
    max_retries: int = 2,
) -> float:
    if isinstance(contexts, list):
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    if not context_str.strip():
        return 0.0

    prompt = prompts.EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str,
        evidence=reference_evidence,
    )

    classifications = await _get_classifications(prompt, llm, max_retries)

    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    return np.nan


async def _get_classifications(
    prompt: str,
    llm: Any,
    max_retries: int,
) -> list[dict[str, Any]]:
    for _ in range(max_retries):
        try:
            response = await llm.chat(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.0
            )
            classifications = await parse_with_fallbacks(
                response.content, key="classifications", llm=llm
            )
            return _validate_classifications(classifications)
        except Exception:
            continue
    return []


def _validate_classifications(classifications: list) -> list[dict[str, Any]]:
    valid = []
    for item in classifications:
        try:
            if (
                isinstance(item, dict)
                and "statement" in item
                and "reason" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "reason": str(item["reason"]),
                    "attributed": int(item["attributed"]),
                })
        except (TypeError, ValueError):
            continue
    return valid
