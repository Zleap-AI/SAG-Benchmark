"""GraphRAG adapter — migrated from external/judge/scripts/convert_to_predictions.py.

Dual-format auto-detection:

  Format A (new benchmark):
    qa_full/qa_results.json → per_example[*].{question,predicted_answer,gold_answers}
    search_results_full.json → [{question, retrieved_docs}]

  Format B (legacy sag-benchmark Step_3):
    evaluation/emf1_graphrag_{ds}.json → per_example[*]
    response/graphrag_{ds}_result.json → [{question, retrieved_docs}]

Always aligned by question text — never by question_index (1-based vs 0-based mismatch).
"""

import json
import os
from typing import Any


def graphrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    # ---- QA results ----
    pe = []
    qa_path = os.path.join(run_dir, "qa_full", "qa_results.json")
    if not os.path.exists(qa_path):
        qa_path = os.path.join(run_dir, "qa_passage", "qa_results.json")
    if os.path.exists(qa_path):
        with open(qa_path, encoding="utf-8") as f:
            pe = json.load(f).get("per_example", [])
    else:
        eval_path = os.path.join(run_dir, "evaluation", f"emf1_graphrag_{dataset_name}.json")
        if os.path.exists(eval_path):
            with open(eval_path, encoding="utf-8") as f:
                pe = json.load(f).get("per_example", [])
        else:
            raise FileNotFoundError(
                f"Not found: qa_results.json (qa_full/ / qa_passage/)"
                f" or evaluation/emf1_graphrag_{dataset_name}.json"
            )

    # ---- retrieval context (question-text aligned) ----
    ctx_by_q: dict[str, str] = {}
    # Format A: search_results_full.json
    sr_path = os.path.join(run_dir, "search_results_full.json")
    if os.path.exists(sr_path):
        with open(sr_path, encoding="utf-8") as f:
            for r in json.load(f):
                q = r.get("question", "").strip()
                docs = r.get("retrieved_docs", []) or []
                ctx_by_q[q] = "\n\n".join(docs)
    # Format B: response/graphrag_{ds}_result.json
    ret_path = os.path.join(run_dir, "response", f"graphrag_{dataset_name}_result.json")
    if os.path.exists(ret_path) and not ctx_by_q:
        with open(ret_path, encoding="utf-8") as f:
            for r in json.load(f):
                q = r.get("question", "").strip()
                docs = r.get("retrieved_docs", []) or []
                if not q:
                    qidx = r.get("question_index")
                    q = pe[qidx - 1]["question"].strip() if qidx and qidx - 1 < len(pe) else ""
                if q:
                    ctx_by_q[q] = "\n\n".join(docs)

    rows = []
    for ex in pe:
        q = ex.get("question", "").strip()
        rows.append({
            "question": q,
            "context": ctx_by_q.get(q, ""),
            "generated_answer": ex.get("predicted_answer", "") or "",
            "ground_truth": ex.get("gold_answers", []),
        })

    matched_ctx = sum(1 for r in rows if r["context"].strip())
    print(f"    context matched: {matched_ctx}/{len(rows)} (by question text)")
    return rows
