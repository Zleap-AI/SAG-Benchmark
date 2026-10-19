"""Verify the retained best full-run results without any model calls."""

import hashlib
import json
from pathlib import Path

from benchmark_data import DatasetLoader
from run_sag2_benchmark import recall_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "results/agentic"
    manifest = json.loads((folder / "provenance.json").read_text())
    checked = {}
    for name, hashes in manifest["datasets"].items():
        target = folder / name
        for file, key in [
            ("questions.jsonl", "questions_sha256"),
            ("summary.json", "summary_sha256"),
        ]:
            assert hashlib.sha256((target / file).read_bytes()).hexdigest() == hashes[key]
        corpus = ROOT / "dataset" / f"{name}_corpus.json"
        assert hashlib.sha256(corpus.read_bytes()).hexdigest() == hashes["corpus_sha256"]
        questions = DatasetLoader(name, ROOT / "dataset").get_question_records()
        rows = [json.loads(line) for line in (target / "questions.jsonl").read_text().splitlines()]
        assert len(rows) == len(questions) == 1000
        summary = json.loads((target / "summary.json").read_text())
        for row, question in zip(rows, questions, strict=True):
            assert row["dataset_sample_id"] == question["id"]
            assert row["question"] == question["question"]
            metrics = recall_metrics(question["gold_docs"], row["retrieved_docs"], [1, 2, 5, 10])
            assert metrics == row["metrics"]
        for metric in summary["metrics"]:
            assert (
                abs(
                    sum(row["metrics"][metric] for row in rows) / len(rows)
                    - summary["metrics"][metric]
                )
                < 1e-12
            )
        assert sum(row["metrics"]["Recall@10"] == 1 for row in rows) == summary["complete_at_10"]
        checked[name] = {"questions": len(rows), "metrics": summary["metrics"]}
    print(json.dumps({"status": "verified", "datasets": checked}, indent=2))


if __name__ == "__main__":
    main()
