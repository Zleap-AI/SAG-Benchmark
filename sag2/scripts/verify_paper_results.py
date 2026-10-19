"""Recalculate the retained paper results without model or network requests."""

import json
import math
from pathlib import Path

from benchmark_data import DatasetLoader

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def check(actual, expected, label):
    if actual.keys() != expected.keys():
        raise ValueError(f"Metric fields differ: {label}")
    for name, value in expected.items():
        if not math.isclose(actual[name], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Metric differs: {label} {name}: {actual[name]} != {value}")


def metrics(gold, retrieved, *, all_evidence=True):
    gold = set(gold)
    retrieved = list(dict.fromkeys(retrieved))
    result = {}
    for k in (1, 2, 5, 10):
        found = gold.intersection(retrieved[:k])
        result[f"Recall@{k}"] = len(found) / len(gold)
        if all_evidence:
            result[f"AllEvidence@{k}"] = float(found == gold)
    return result


def mean(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    latest = read_json(RESULTS / "paper/latest-results.json")
    study = read_json(RESULTS / "paper/controlled-study.json")
    sources = {}
    for name in ("musique", "2wikimultihopqa", "hotpotqa"):
        records = DatasetLoader(name, ROOT / "dataset").get_question_records()
        sources[name] = {row["id"]: row for row in records}
        if len(sources[name]) != 1000:
            raise ValueError(f"Incomplete source dataset: {name}")

    verified = {}
    for name, expected in latest["benchmarks"].items():
        folder = RESULTS / ("controls/sag2_h2" if name == "musique" else f"benchmarks/{name}")
        rows = read_rows(folder / "questions.jsonl")
        ids = [row.get("dataset_sample_id", row.get("id")) for row in rows]
        if len(ids) != 1000 or set(ids) != sources[name].keys():
            raise ValueError(f"Question identity mismatch: {name}")
        measured = []
        for identifier, row in zip(ids, rows):
            result = metrics(sources[name][identifier]["gold_docs"], row["retrieved_docs"])
            check(
                {key: result[key] for key in row["metrics"]}, row["metrics"], f"{name}/{identifier}"
            )
            measured.append(result)
        check(mean(measured), expected["metrics"], name)
        verified[name] = {"questions": len(rows), "metrics": mean(measured)}

    for name, expected in study["variants"].items():
        rows = read_rows(RESULTS / "controls" / name / "questions.jsonl")
        ids = [row["dataset_sample_id"] for row in rows]
        if len(ids) != 1000 or set(ids) != sources["musique"].keys():
            raise ValueError(f"Incomplete control: {name}")
        for row in rows:
            source = sources["musique"][row["dataset_sample_id"]]
            if row["question"] != source["question"] or row["gold_docs"] != source["gold_docs"]:
                raise ValueError(f"Control question changed: {name}/{row['dataset_sample_id']}")
            check(metrics(row["gold_docs"], row["retrieved_docs"]), row["metrics"], name)
            if row["successful_scoring_requests"] > row["successful_request_cap"]:
                raise ValueError(f"Successful request budget exceeded: {name}")
        for group, group_expected in expected["groups"].items():
            selected = (
                rows if group == "all" else [r for r in rows if str(r["problem_hops"]) == group]
            )
            if len(selected) != group_expected["n"]:
                raise ValueError(f"Control group size changed: {name}/{group}")
            check(
                mean([r["metrics"] for r in selected]), group_expected["metrics"], f"{name}/{group}"
            )
        for stage, stage_expected in expected["stages"].items():
            selected = [row["stage_results"][stage]["metrics"] for row in rows]
            check(mean(selected), stage_expected["metrics"], f"{name}/{stage}")
        for field in ("candidate_judgments", "successful_scoring_requests"):
            if sum(row[field] for row in rows) != expected[field]:
                raise ValueError(f"Control accounting mismatch: {name}/{field}")
        verified[name] = {"questions": len(rows), "metrics": mean([r["metrics"] for r in rows])}

    rows = read_rows(RESULTS / "enterprise/questions.jsonl")
    official = read_json(RESULTS / "enterprise/questions.json")
    eligible = {q["question_id"]: q for q in official if q["expected_doc_ids"]}
    if (
        len(official) != 500
        or len(rows) != 470
        or {r["question_id"] for r in rows} != eligible.keys()
    ):
        raise ValueError("EnterpriseRAG identity or coverage mismatch")
    for row in rows:
        source = eligible[row["question_id"]]
        if row["question"] != source["question"] or set(row["gold_document_ids"]) != set(
            source["expected_doc_ids"]
        ):
            raise ValueError(f"EnterpriseRAG question changed: {row['question_id']}")
        for method, field in (("sag2", "retrieved_source_ids"), ("bm25", "bm25_source_ids")):
            check(
                metrics(row["gold_document_ids"], row[field], all_evidence=False),
                row["metrics"][method],
                row["question_id"],
            )
    enterprise = read_json(RESULTS / "paper/enterprise-results.json")
    for group, expected in enterprise["metrics"].items():
        selected = rows if group == "overall" else [r for r in rows if r["question_type"] == group]
        check(mean([r["metrics"]["sag2"] for r in selected]), expected, f"enterprise/{group}")
    verified["enterprise"] = {"questions": 470, "excluded_without_gold": 30}
    report = {"status": "passed", "network_requests": 0, "verified": verified}
    (RESULTS / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Verified: three 1,000-question datasets, eight controls, and 470 EnterpriseRAG questions. No model requests."
    )


if __name__ == "__main__":
    main()
