"""Answer Correctness aligned with GraphRAG-Benchmark/Evaluation/metrics/answer_accuracy.py.

Pure LLM-judge metric: generates statements, classifies TP/FP/FN, computes F-beta.
No embedding/similarity component.

Unlike the reference, an unusable classification response raises instead of scoring
0.0, so a parse failure is recorded as a failed sample rather than being averaged in
as a genuinely wrong answer.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.evaluation.judge import prompts
from pipeline.evaluation.judge.parser import parse_with_fallbacks
from pipeline.exceptions import LLMResponseError


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason] = []
    FP: list[StatementsWithReason] = []
    FN: list[StatementsWithReason] = []


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: Any,
    beta: float = 1.0,
    return_intermediate: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    answer_statements = await _generate_statements(llm, question, answer)
    gt_statements = await _generate_statements(llm, question, ground_truth)
    factuality_result = await _calculate_factuality(
        llm,
        question,
        answer_statements,
        gt_statements,
        beta,
        return_intermediate=return_intermediate,
    )
    if return_intermediate:
        score, classification = factuality_result
        intermediate = {
            "answer_statements": answer_statements,
            "gt_statements": gt_statements,
            "classification": classification,
        }
        return (score, intermediate)
    return float(factuality_result)


async def _generate_statements(llm: Any, question: str, answer: str) -> list[str]:
    prompt = prompts.STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    response = await llm.chat(
        [LLMMessage(role=LLMRole.USER, content=prompt)],
        temperature=0.0,
        seed=42,
    )
    parsed = await parse_with_fallbacks(response.content)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        for key in ["statements", "answers", "items", "list", "output", "result"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
        return [str(v) for v in parsed.values()]
    return [str(parsed)]


async def _calculate_factuality(
    llm: Any,
    question: str,
    answer_stmts: list[str],
    gt_stmts: list[str],
    beta: float,
    return_intermediate: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    if not answer_stmts and not gt_stmts:
        empty = {"TP": [], "FP": [], "FN": []}
        return (1.0, empty) if return_intermediate else 1.0

    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in prompts.CORRECTNESS_EXAMPLES
    )

    prompt = prompts.CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts),
    )
    response = await llm.chat(
        [LLMMessage(role=LLMRole.USER, content=prompt)],
        temperature=0.0,
        seed=42,
    )

    try:
        classification = ClassificationWithReason(**json.loads(response.content))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise LLMResponseError(f"Correctness classification response is unusable: {exc}") from exc

    tp = len(classification.TP)
    fp = len(classification.FP)
    fn = len(classification.FN)
    score = fbeta_score(tp, fp, fn, beta)
    intermediate = {
        "TP": [s.model_dump() for s in (classification.TP or [])],
        "FP": [s.model_dump() for s in (classification.FP or [])],
        "FN": [s.model_dump() for s in (classification.FN or [])],
    }
    return (score, intermediate) if return_intermediate else score
