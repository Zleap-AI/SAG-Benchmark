"""HyperGraphRAG adapter — migrated from external/judge/scripts/convert_to_predictions.py.

Reads response/hybrid_<ds>_result.json.
"""

import json
import os
from typing import Any


def hypergraphrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    resp_path = os.path.join(run_dir, "response", f"hybrid_{dataset_name}_result.json")
    if not os.path.exists(resp_path):
        raise FileNotFoundError(f"Not found: {resp_path}")

    with open(resp_path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        {
            "question": item["query"],
            "context": item.get("context", ""),
            "generated_answer": item.get("pred_answer", ""),
        }
        for item in data
    ]
