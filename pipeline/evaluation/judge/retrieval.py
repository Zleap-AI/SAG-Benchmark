"""Retrieval Judge evaluation — migrated from external/judge/Evaluation/retrieval_eval.py.

Evaluates context_relevancy and evidence_recall. Evidence is loaded LIVE from
the raw dataset. No embedding/Ragas/Datasets/LangChain dependency.
"""

import asyncio
import json
import math
import os
from typing import Any

from pipeline.evaluation.judge.checkpoint import (
    _select_rerun_ids,
    atomic_write_json,
    load_checkpoint,
    merge_partial_results,
)
from pipeline.evaluation.judge.dataset_io import load_evidence_map
from pipeline.evaluation.judge.metrics import context_relevance, evidence_recall
from pipeline.utils import get_logger

logger = get_logger(__name__)

ALL_RETRIEVAL_METRICS = {"context_relevancy", "evidence_recall"}


def _is_graph_based_context(context: str) -> bool:
    """Return whether context uses the structured graph-section format."""
    return isinstance(context, str) and context.strip().startswith("-----")


def _preprocess_context(context: Any, context_top_k: int = 5) -> list[str]:
    """Preserve graph context and limit plain passage context to top-k chunks."""
    if context is None:
        return []
    if isinstance(context, list):
        context = "\n\n".join(str(item) for item in context if item)
    if not isinstance(context, str) or not context.strip():
        return []
    if _is_graph_based_context(context) or context_top_k <= 0:
        return [context]
    chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    return chunks[:context_top_k]


def _average_scores(scores: dict[str, list[float]]) -> dict[str, float]:
    """Average metric values while excluding failed-sample NaNs."""
    averages: dict[str, float] = {}
    for metric, values in scores.items():
        valid = [
            value
            for value in values
            if not (isinstance(value, float) and math.isnan(value))
        ]
        averages[metric] = sum(valid) / len(valid) if valid else float("nan")
    return averages


async def evaluate_sample(
    question: str,
    contexts: list[str],
    evidences: list[str],
    llm: Any,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval metrics for a single sample."""
    if metrics is None:
        metrics = list(ALL_RETRIEVAL_METRICS)

    tasks = {}
    if "context_relevancy" in metrics:
        tasks["context_relevancy"] = context_relevance.compute_context_relevance(
            question, contexts, llm
        )
    if "evidence_recall" in metrics:
        tasks["evidence_recall"] = evidence_recall.compute_evidence_recall(
            question, contexts, evidences, llm
        )

    task_results = await asyncio.gather(*tasks.values())
    results = {}
    for metric, result in zip(tasks.keys(), task_results):
        results[metric] = result

    return results


async def run_retrieval_eval(
    data_file: str,
    llm: Any,
    output_file: str | None = None,
    detailed_output: bool = True,
    max_concurrent: int = 4,
    only_metrics: list[str] | None = None,
    force: bool = False,
    dataset: str | None = None,
    dataset_dir: str = "dataset",
    context_top_k: int = 5,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """Run full retrieval evaluation pipeline."""
    detailed_output = True
    with open(data_file, encoding="utf-8") as f:
        file_data = json.load(f)

    eval_metrics = only_metrics if only_metrics else list(ALL_RETRIEVAL_METRICS)
    if only_metrics:
        illegal = [m for m in only_metrics if m not in ALL_RETRIEVAL_METRICS]
        if illegal:
            logger.error("Invalid metrics: %s", illegal)
            raise SystemExit(1)

    # Output path
    judge_model = getattr(llm.config, "model", "judge")
    out_dir = os.path.dirname(output_file) if output_file else "."
    out_name = os.path.basename(output_file) if output_file else "results.json"
    judge_dir = os.path.join(out_dir, f"LlmJudge_{judge_model}")
    final_path = os.path.join(judge_dir, out_name)
    ckpt_path = final_path + ".partial"

    # Resume
    existing: dict[str, Any] = {}
    resume_mode = False
    if not force and os.path.exists(final_path):
        existing = load_checkpoint(final_path) or {}
        if existing:
            resume_mode = True
            logger.info("Found existing results, resuming")

    # Evidence maps (live from raw dataset)
    forced_ds = dataset
    evidence_maps: dict[str, dict[int, list[str]]] = {}

    def _evidence_for(item: dict[str, Any]) -> list[str]:
        ds = forced_ds or item.get("source", "")
        if ds and ds not in evidence_maps:
            try:
                evidence_maps[ds] = load_evidence_map(ds, dataset_dir)
                n_nonempty = sum(
                    1 for v in evidence_maps[ds].values() if v
                )
                logger.info(
                    "Loaded evidence from dataset/%s.json: %d/%d samples",
                    ds, n_nonempty, len(evidence_maps[ds]),
                )
            except Exception as e:
                logger.warning(
                    "Could not load dataset/%s.json (%s); falling back to predictions",
                    ds, e,
                )
                evidence_maps[ds] = {}
        emap = evidence_maps.get(ds, {})
        if emap:
            return emap.get(item.get("id"), item.get("evidence", []))
        return item.get("evidence", [])

    # Group by question type
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        grouped.setdefault(q_type, []).append(item)

    all_results: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    for question_type, group_items in grouped.items():
        logger.info("Evaluating: %s (%d samples)", question_type, len(group_items))

        rerun_ids: set[int] | None = None
        if resume_mode and question_type in existing:
            old_detailed = existing[question_type].get("detailed", [])
            all_ids_q = {it["id"] for it in group_items}
            rerun_ids = _select_rerun_ids(old_detailed, all_ids_q, force_all=bool(only_metrics))
            if rerun_ids:
                group_items = [it for it in group_items if it["id"] in rerun_ids]
                logger.info("  Re-evaluating %d samples", len(rerun_ids))
            else:
                logger.info("  All complete, skipping")
                all_results[question_type] = existing[question_type]
                continue

        if num_samples and num_samples < len(group_items):
            group_items = group_items[:num_samples]

        detailed_entries: list[dict[str, Any]] = []
        all_scores: dict[str, list[float]] = {}

        async def _eval_one(item: dict[str, Any]) -> dict[str, Any]:
            try:
                async with semaphore:
                    ctx = _preprocess_context(
                        item.get("context", item.get("contexts", [])), context_top_k
                    )
                    ev = _evidence_for(item)
                    result = await evaluate_sample(
                        question=item["question"],
                        contexts=ctx,
                        evidences=ev,
                        llm=llm,
                        metrics=eval_metrics,
                    )
                return {"id": item["id"], "item": item, "result": result}
            except Exception as exc:
                logger.error(
                    "Sample %s failed with %s",
                    item.get("id", "?"),
                    type(exc).__name__,
                )
                return {"id": item.get("id"), "item": item, "error": exc}

        tasks = [_eval_one(it) for it in group_items]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            entry = await coro
            item = entry["item"]
            result = entry.get("result") or {
                metric: float("nan") for metric in eval_metrics
            }
            completed += 1

            detailed_entries.append({
                "id": item["id"],
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "generated_answer": item.get("answer", item.get("generated_answer", "")),
                "contexts": item.get("contexts", item.get("context", [])),
                "metrics": result,
                "llm_intermediate": {},
            })
            for m, v in result.items():
                all_scores.setdefault(m, []).append(v)

            if completed % max(1, len(group_items) // 10) == 0:
                logger.info("  Progress: %d/%d", completed, len(group_items))

            if completed % 5 == 0:
                try:
                    avg = _average_scores(all_scores)
                    atomic_write_json(
                        {question_type: {"average_scores": avg, "detailed": detailed_entries}},
                        ckpt_path,
                    )
                except Exception as e:
                    logger.warning("Checkpoint write skipped: %s", e)

        avg = _average_scores(all_scores)
        q_result = {"average_scores": avg, "detailed": detailed_entries}

        if resume_mode and question_type in existing and rerun_ids:
            q_result = merge_partial_results(
                existing, {question_type: q_result}, question_type, rerun_ids, only_metrics
            )

        all_results[question_type] = q_result

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    if output_file:
        os.makedirs(judge_dir, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", final_path)

    return all_results
