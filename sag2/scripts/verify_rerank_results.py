"""Recompute the retained reranker ablations without inference or network access."""

import hashlib
import json
import math
from pathlib import Path

from benchmark_data import DatasetLoader

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "results/rerank"
    summary = json.loads((folder / "summary.json").read_text())
    verified, aggregate_only = 0, []
    for name, entry in summary["experiments"].items():
        if entry["records"] is None:
            aggregate_only.append(name)
            continue
        source = folder / entry["records"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["records_sha256"]:
            raise ValueError(f"Reranker records changed: {name}")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        questions = {
            r["id"]: r["question"]
            for r in DatasetLoader(entry["dataset"], ROOT / "dataset").get_question_records()
        }
        if len(rows) != 1000 or {r["dataset_sample_id"] for r in rows} != set(questions):
            raise ValueError(f"Incomplete question set: {name}")
        for row in rows:
            if row["question"] != questions[row["dataset_sample_id"]]:
                raise ValueError(f"Question identity changed: {name}")
            gold, retrieved = set(row["gold_document_ids"]), row["retrieved_document_ids"]
            if len(retrieved) != len(set(retrieved)):
                raise ValueError(f"Duplicate retrieved document: {name}")
            for k in (1, 2, 5, 10):
                found = gold.intersection(retrieved[:k])
                if not math.isclose(
                    len(found) / len(gold), row["metrics"][f"Recall@{k}"], abs_tol=1e-12
                ):
                    raise ValueError(f"Recall mismatch: {name}/{row['question_index']}")
                if int(found == gold) != row["metrics"][f"AllEvidence@{k}"]:
                    raise ValueError(f"AllEvidence mismatch: {name}/{row['question_index']}")
        for key, expected in entry["metrics"].items():
            if not math.isclose(
                sum(r["metrics"][key] for r in rows) / len(rows), expected, abs_tol=1e-12
            ):
                raise ValueError(f"Aggregate mismatch: {name}/{key}")
        verified += len(rows)
    print(
        f"Verified {verified} reranker question records. Aggregate-only historical results: {', '.join(aggregate_only)}. No model requests."
    )


if __name__ == "__main__":
    main()
