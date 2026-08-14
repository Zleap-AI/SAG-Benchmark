"""HippoRAG2 adapter — migrated from external/judge/scripts/convert_to_predictions.py.

Reads <llm>_<emb>/qa_result/qa_results_latest.json.
"""

import json
import os
from typing import Any


def hipporag2_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    qa_file = None
    for sub in sorted(os.listdir(run_dir), reverse=True):
        sub_path = os.path.join(run_dir, sub, "qa_result", "qa_results_latest.json")
        if os.path.isfile(sub_path):
            qa_file = sub_path
            break
    if not qa_file:
        raise FileNotFoundError(
            f"Could not find <llm>_<emb>/qa_result/qa_results_latest.json in {run_dir}"
        )

    with open(qa_file, encoding="utf-8") as f:
        data = json.load(f)

    return [
        {
            "question": item["question"],
            "context": "\n\n".join(item.get("docs", [])),
            "generated_answer": item.get("answer", ""),
            "ground_truth": item.get("gold_answers", []),
        }
        for item in data.get("results", [])
    ]
