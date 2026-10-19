"""Evidence Recall aligned with GraphRAG-Benchmark/Evaluation/metrics/evidence_recall.py.

Measures what percentage of reference evidence is supported by the context.
Evidence is loaded LIVE from the raw dataset, not from predictions.

Matches the authoritative implementation: no context chunking, `statement`
field classification with loose validation, and `sum(attributed) / len(valid)`
denominator (valid = classifications that pass validation).
"""

from typing import Any

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.errors import MetricResultError
from pipeline.evaluation.judge.parser import parse_with_fallbacks
from pipeline.exceptions import LLMResponseError


async def compute_evidence_recall(
    question: str,
    contexts: str | list[str],
    reference_evidence: list[str],
    llm: Any,
) -> float:
    if isinstance(contexts, list):
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    if not context_str.strip():
        return 0.0
    if not reference_evidence:
        raise MetricResultError(
            "Evidence recall requires at least one canonical evidence statement"
        )

    prompt = prompts.EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str,
        evidence=reference_evidence,
    )
    classifications = await _get_classifications(prompt, llm)
    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    raise LLMResponseError("Evidence recall returned no valid classifications")


async def _get_classifications(
    prompt: str,
    llm: Any,
) -> list[dict[str, Any]]:
    response = await llm.chat(
        [LLMMessage(role=LLMRole.USER, content=prompt)],
        temperature=0.0,
        seed=42,
    )
    classifications = await parse_with_fallbacks(
        response.content,
        key="classifications",
    )
    return _validate_classifications(classifications)


def _validate_classifications(classifications: Any) -> list[dict[str, Any]]:
    """Loose validation matching the reference implementation.

    Keeps every classification that has `statement`, `reason`, and an
    `attributed` flag in {0, 1}; drops (not raises on) everything else.
    """
    valid: list[dict[str, Any]] = []
    for item in classifications:
        try:
            if (
                isinstance(item, dict)
                and "statement" in item
                and "reason" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append(
                    {
                        "statement": str(item["statement"]),
                        "reason": str(item["reason"]),
                        "attributed": int(item["attributed"]),
                    }
                )
        except (TypeError, ValueError):
            continue
    return valid
