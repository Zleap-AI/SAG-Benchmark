"""Base adapter interface and shared utilities.

Migrated from external/judge/scripts/convert_to_predictions.py.
"""

import json
import os
import re
from typing import Any


def _is_nonempty(val: Any) -> bool:
    """Return whether a scalar or multi-gold value contains meaningful text."""
    if val is None:
        return False
    if isinstance(val, list):
        return any(v.strip() for v in val if isinstance(v, str))
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)


def find_latest_run(input_root: str, dataset: str) -> str | None:
    """Find the latest run directory under input_root.

    Supports two modes:
      1. input_root/<dataset>/ → find latest subdir (e.g. SAG timestamp dirs)
      2. input_root/<dataset>_* prefix match (e.g. musique_20260728_014955)
    Mode 2 takes priority — it's a pure data directory.
    """
    # Prefix match first (pure data dirs)
    prefix_match = None
    if os.path.isdir(input_root):
        candidates = sorted(
            [d for d in os.listdir(input_root)
             if os.path.isdir(os.path.join(input_root, d))
             and d.startswith(f"{dataset}_")
             and not d.startswith("LlmJudge_")],
            reverse=True,
        )
        if candidates:
            prefix_match = os.path.join(input_root, candidates[0])

    ds_dir = os.path.join(input_root, dataset)
    if not os.path.isdir(ds_dir):
        if prefix_match and os.path.isdir(prefix_match):
            ds_dir = prefix_match
        if not os.path.isdir(ds_dir):
            return None
    subdirs = sorted(
        (
            os.path.join(ds_dir, d)
            for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d))
            and not d.startswith("LlmJudge_")
        ),
        reverse=True,
    )
    for subdir in subdirs:
        if _has_data_files(subdir):
            return subdir
    if _has_data_files(ds_dir):
        return ds_dir
    if prefix_match and os.path.isdir(prefix_match) and _has_data_files(prefix_match):
        print(f"    i {dataset}/ exists but no data files, falling back to prefix match: {prefix_match}")
        return prefix_match
    return None


def _has_data_files(dirpath: str) -> bool:
    """Check if directory contains expected data files."""
    if not os.path.isdir(dirpath):
        return False
    if os.path.isdir(os.path.join(dirpath, "response")):
        return True
    if any(
        f.startswith("results_") and f.endswith(".json")
        for f in os.listdir(dirpath)
    ):
        return True
    # graphrag: search_results_full.json at top level
    if os.path.exists(os.path.join(dirpath, "search_results_full.json")):
        return True
    # HyperGraphRAG direct: hybrid_<ds>_result.json at top level
    if any(f.startswith("hybrid_") and f.endswith("_result.json") for f in os.listdir(dirpath)):
        return True
    if os.path.isdir(os.path.join(dirpath, "qa_result")):
        return True
    for sub in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, sub, "qa_result")):
            return True
    if os.path.isdir(os.path.join(dirpath, "evaluation")):
        return True
    return False


def convert_one(
    project: str,
    dataset: str,
    input_root: str,
    dataset_dir: str,
    out_root: str,
    adapters: dict[str, Any],
) -> str | None:
    """Convert a single dataset and write predictions JSON.

    Returns the output file path or None on failure.
    """
    run_dir = find_latest_run(input_root, dataset)
    if not run_dir:
        run_dir = os.path.join(input_root, dataset)
    if not _has_data_files(run_dir):
        parent = os.path.dirname(run_dir)
        if _has_data_files(parent):
            run_dir = parent
    if not os.path.isdir(run_dir) or not _has_data_files(run_dir):
        print(f"  Could not find data directory for {dataset}")
        return None

    adapter = adapters.get(project)
    if adapter is None:
        raise NotImplementedError(f"No adapter registered for project '{project}'")
    rows = adapter(run_dir, dataset)

    # Enrich with ground_truth and id from raw dataset
    # Auto-match dataset name: exact → strip timestamp suffix → strip trailing digits
    raw_path = os.path.join(dataset_dir, f"{dataset}.json")
    if not os.path.exists(raw_path):
        for pattern in [r'_\d{8}_\d{6}$', r'\d+$']:
            base = re.sub(pattern, '', dataset)
            if base != dataset:
                alt = os.path.join(dataset_dir, f"{base}.json")
                if os.path.exists(alt):
                    raw_path = alt
                    break
    gt_map: dict[int, str] = {}
    q_to_id: dict[str, int] = {}
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        for i, sample in enumerate(raw):
            gt_map[i] = sample.get("answer", "")
            q_to_id[sample["question"].strip()] = i

    id_by_q = 0
    for r in rows:
        if "id" not in r:
            q = r["question"].strip()
            r["id"] = q_to_id.get(q, id_by_q)
            id_by_q += 1
        r["source"] = dataset
        if not r.get("ground_truth"):
            r["ground_truth"] = gt_map.get(r["id"], "")
        r["question_type"] = "qa"
        r["evidence"] = ""

    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"predictions_{dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {out_path} ({len(rows)} rows)")
    return out_path
