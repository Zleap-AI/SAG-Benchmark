"""Shared atomic checkpoint write for Judge evaluation.

Replaces the copy-pasted checkpoint logic in generation_eval.py and retrieval_eval.py.
"""

import json
import math
import os
import tempfile
from typing import Any

from pipeline.utils import get_logger

logger = get_logger(__name__)


def atomic_write_json(data: Any, target_path: str) -> None:
    """Write JSON atomically: tmp -> flush -> os.replace.

    If the write fails, the original file is untouched.
    """
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".judge_ckpt_", dir=target_dir
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _entry_has_nan(entry: dict[str, Any]) -> bool:
    """Check whether any metric score in a detailed entry is NaN."""
    metrics = entry.get("metrics", {})
    return any(
        isinstance(v, float) and math.isnan(v) for v in metrics.values()
    )


def _select_rerun_ids(
    old_detailed: list[dict[str, Any]],
    all_ids: set[int],
    force_all: bool = False,
) -> set[int]:
    """Determine which sample ids need re-evaluation.

    Returns NaN entries, missing entries, and optionally all entries.
    """
    if force_all:
        return all_ids
    nan_ids = {ent["id"] for ent in old_detailed if _entry_has_nan(ent)}
    done_ids = {ent["id"] for ent in old_detailed}
    missing_ids = all_ids - done_ids
    return nan_ids | missing_ids


def load_checkpoint(path: str) -> dict[str, Any] | None:
    """Load an existing checkpoint file.

    Returns None if the file doesn't exist or is corrupted.
    A corrupted checkpoint is NOT deleted — only a warning is logged.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Checkpoint %s is corrupted (not deleting): %s", path, e
        )
        return None


def merge_partial_results(
    existing: dict[str, Any],
    new_results: dict[str, Any],
    question_type: str,
    rerun_ids: set[int],
    only_metrics: list[str] | None,
) -> dict[str, Any]:
    """Merge new evaluation results into existing results for a question_type.

    - Entries in rerun_ids get replaced.
    - If only_metrics is set, only those metrics are updated per entry.
    - Other entries and metrics are preserved.
    """
    old_by_id = {
        d["id"]: d for d in existing.get(question_type, {}).get("detailed", [])
    }
    result_section = new_results.get(question_type, new_results)
    for d in result_section.get("detailed", []):
        oid = d["id"]
        if rerun_ids and oid not in rerun_ids:
            continue
        if only_metrics:
            if oid in old_by_id:
                old = old_by_id[oid]
                for m in only_metrics:
                    if m in d.get("metrics", {}):
                        old["metrics"][m] = d["metrics"][m]
            else:
                new_entry = dict(d)
                new_entry["metrics"] = {
                    metric: value
                    for metric, value in d.get("metrics", {}).items()
                    if metric in only_metrics
                }
                old_by_id[oid] = new_entry
        else:
            old_by_id[oid] = d

    merged_detailed = list(old_by_id.values())
    all_metrics: set[str] = set()
    for d_entry in merged_detailed:
        all_metrics.update(d_entry.get("metrics", {}).keys())

    new_avg: dict[str, float] = {}
    for m in all_metrics:
        vals = [
            d_entry["metrics"][m]
            for d_entry in merged_detailed
            if m in d_entry.get("metrics", {})
            and not (
                isinstance(d_entry["metrics"][m], float)
                and math.isnan(d_entry["metrics"][m])
            )
        ]
        new_avg[m] = sum(vals) / len(vals) if vals else float("nan")

    return {"average_scores": new_avg, "detailed": merged_detailed}
