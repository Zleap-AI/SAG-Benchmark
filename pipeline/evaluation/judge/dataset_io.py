"""Dataset I/O — load predictions and raw datasets without `datasets.Dataset`."""

import json
import os
from typing import Any


def load_predictions(path: str) -> list[dict[str, Any]]:
    """Load predictions JSON file (list of samples)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_dataset(dataset_name: str, dataset_dir: str) -> list[dict[str, Any]]:
    """Load a raw dataset JSON file.

    Args:
        dataset_name: e.g. "hotpotqa"
        dataset_dir: directory containing {dataset_name}.json

    Returns:
        List of raw samples with question, answer, evidence etc.
    """
    path = os.path.join(dataset_dir, f"{dataset_name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_evidence_map(
    dataset_name: str, dataset_dir: str
) -> dict[int, list[str]]:
    """Build an id -> [evidence_strings] map from the raw dataset.

    Evidence may be stored directly (``evidence``/``evidences``) or as
    HotpotQA ``supporting_facts`` references into the sample ``context``.
    Returns a dict keyed by enumerate index (0-based).
    """
    raw = load_raw_dataset(dataset_name, dataset_dir)
    emap: dict[int, list[str]] = {}
    for i, sample in enumerate(raw):
        ev = sample.get("evidence", sample.get("evidences", []))
        if not ev and sample.get("supporting_facts"):
            context_by_title = {
                str(block[0]): block[1]
                for block in sample.get("context", [])
                if isinstance(block, list)
                and len(block) >= 2
                and isinstance(block[1], list)
            }
            ev = []
            for fact in sample["supporting_facts"]:
                if not isinstance(fact, list) or len(fact) < 2:
                    continue
                title, sentence_index = str(fact[0]), fact[1]
                sentences = context_by_title.get(title, [])
                if (
                    isinstance(sentence_index, int)
                    and 0 <= sentence_index < len(sentences)
                ):
                    ev.append(str(sentences[sentence_index]))
        if isinstance(ev, str):
            ev = [ev] if ev.strip() else []
        elif not isinstance(ev, list):
            ev = []
        emap[i] = [str(e) for e in ev if str(e).strip()]
    return emap


def group_by_question_type(
    data: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group prediction items by question_type."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in data:
        q_type = item.get("question_type", "Uncategorized")
        grouped.setdefault(q_type, []).append(item)
    return grouped
