"""Full official EnterpriseRAG recall evaluation, with per-category reports."""

import argparse
import asyncio
import csv
import json
import os
import platform
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_index import ReadOnlyIndex, save  # noqa: E402
from sag2 import JevClient, corpus_fingerprint, search_sag2  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
PARAMETERS = {"candidate_k": 100, "seed_k": 10, "max_hops": 2, "expansion_term_limit": 20}
KS = (1, 2, 5, 10)


def aggregate(questions, results, failures):
    groups = {"overall": questions}
    groups.update(
        {
            kind: [q for q in questions if q["question_type"] == kind]
            for kind in dict.fromkeys(q["question_type"] for q in questions)
        }
    )
    report = {}
    for kind, members in groups.items():
        eligible = [q for q in members if q["expected_doc_ids"]]
        finished = [results[q["question_id"]] for q in eligible if q["question_id"] in results]
        report[kind] = {
            "total_questions": len(members),
            "eligible_questions": len(eligible),
            "completed_questions": len(finished),
            "failed_questions": sum(q["question_id"] in failures for q in eligible),
            "not_applicable_questions": len(members) - len(eligible),
            "recall": {
                method: {
                    f"Recall@{k}": sum(r["metrics"][method][f"Recall@{k}"] for r in finished)
                    / len(finished)
                    if finished
                    else None
                    for k in KS
                }
                for method in ("sag2", "bm25")
            },
        }
    return report


class RecordedJev(JevClient):
    def __init__(self, api_key, directory, activity):
        super().__init__(None, ENDPOINT, MODEL, api_key)
        self.directory, self.activity = directory, activity
        self.request_lock = asyncio.Lock()
        self.round = 0
        self.costs, self.reused = [], 0

    async def request_started(self, session, context, params):
        self.attempt += 1
        self.attempt_started = time.perf_counter()
        with self.activity["lock"]:
            self.activity["http_active"] += 1
            self.activity["http_started"] += 1
            self.activity["peak_http"] = max(
                self.activity["peak_http"], self.activity["http_active"]
            )

    async def request_finished(self, session, context, params):
        save(
            self.round_dir / f"attempt_{self.attempt:05}.json",
            {
                "status": params.response.status,
                "seconds": time.perf_counter() - self.attempt_started,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        with self.activity["lock"]:
            self.activity["http_active"] -= 1
            self.activity["statuses"][str(params.response.status)] += 1

    async def request_exception(self, session, context, params):
        kind = (
            "timeout"
            if isinstance(params.exception, TimeoutError)
            else type(params.exception).__name__
        )
        save(
            self.round_dir / f"attempt_{self.attempt:05}.json",
            {
                "error_type": kind,
                "seconds": time.perf_counter() - self.attempt_started,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        with self.activity["lock"]:
            self.activity["http_active"] -= 1
            self.activity["statuses"][kind] += 1

    async def request(self, payload):
        async with self.request_lock:
            self.round += 1
            self.round_dir = self.directory / f"round_{self.round}"
            self.round_dir.mkdir(parents=True, exist_ok=True)
            saved_payload = self.round_dir / "payload.json"
            if saved_payload.exists():
                if json.loads(saved_payload.read_text()) != payload:
                    raise ValueError("Resume payload differs from the saved request")
            else:
                save(saved_payload, payload)
            response_path = self.round_dir / "response.json"
            if response_path.exists():
                response = json.loads(response_path.read_text())
                self.reused += 1
            else:
                self.attempt = len(list(self.round_dir.glob("attempt_*.json")))
                response = await super().request(payload)
                save(response_path, response)
            self.costs.append(response.get("usage", {}).get("cost"))
            return response


def run_question(question, build, shared, api_key, activity, output):
    index = ReadOnlyIndex(
        Path(build["index_path"]),
        build["index_signature"],
        Path(build["source_directory"]),
        shared[0],
        lexical_statistics=shared[1],
    )

    async def execute():
        client = RecordedJev(api_key, output / "requests" / question["question_id"], activity)
        trace = aiohttp.TraceConfig()
        trace.on_request_start.append(client.request_started)
        trace.on_request_end.append(client.request_finished)
        trace.on_request_exception.append(client.request_exception)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300), trace_configs=[trace]
        ) as session:
            client.session = session
            result = await search_sag2(index, client, question["question"], **PARAMETERS)
        assert sum(row["stage"] == "jev" for row in result["trace"]) <= 3
        gold = set(question["expected_doc_ids"])
        result["metrics"] = {}
        for method, field in (("sag2", "candidates"), ("bm25", "baseline")):
            unique = {}
            for candidate in result[field]:
                unique.setdefault(candidate["source_id"], candidate)
            result[field] = list(unique.values())[:10]
            result["metrics"][method] = {
                f"Recall@{k}": len(gold.intersection(row["source_id"] for row in result[field][:k]))
                / len(gold)
                for k in KS
            }
        result.update(
            question_id=question["question_id"],
            question=question["question"],
            question_type=question["question_type"],
            gold_document_ids=sorted(gold),
            response_costs_usd=client.costs,
            reused_successful_requests=client.reused,
        )
        return result

    try:
        return asyncio.run(execute())
    finally:
        index.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT.parent.parent / "datasets/EnterpriseRAG-Bench"
    )
    parser.add_argument(
        "--index-summary", type=Path, default=ROOT / "output/enterprise/index/summary.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/enterprise/evaluation")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 20:
        parser.error("Concurrency must be between 1 and 20")
    source = args.dataset_dir / "questions.jsonl"
    questions = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(questions) == 500 and len({q["question_id"] for q in questions}) == 500
    eligible = [q for q in questions if q["expected_doc_ids"]]
    assert len(eligible) == 470
    build = json.loads(args.index_summary.read_text())
    env = dotenv_values(args.env_file)
    api_key = (
        env.get("OPENROUTER_API_KEY")
        or env.get("JEV_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("JEV_API_KEY")
    )
    if not api_key:
        raise ValueError("OpenRouter API key is missing")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "algorithm": "SAG2",
        "provider": "OpenRouter",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "question_type": "noul",
        "jev_batching_policy": "sequential_candidate_batches_64000_tokens",
        "concurrency": args.concurrency,
        "parameters": PARAMETERS,
        "scope": "Official 500-question main set; 470 with recall labels; no answer generation",
        "question_types": dict(Counter(q["question_type"] for q in questions)),
        "question_file_sha256": corpus_fingerprint(source),
        "index_signature": build["index_signature"],
        "index": build["index_path"],
        "source_sha256": {
            p.name: corpus_fingerprint(p)
            for p in [
                Path(__file__),
                ROOT / "scripts/sag2.py",
                ROOT / "scripts/enterprise_index.py",
            ]
        },
    }
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume configuration differs")
    else:
        save(output / "config.json", config)
        save(output / "questions.json", questions)
        save(output / "not_applicable.json", [q for q in questions if not q["expected_doc_ids"]])
    save(
        output / "runtime.json",
        {
            "host": platform.node(),
            "pid": os.getpid(),
            "python": sys.executable,
            "started_at": datetime.now(UTC).isoformat(),
            "endpoint": ENDPOINT,
            "concurrency": args.concurrency,
        },
    )
    save(output / "progress.json", {"phase": "loading_index", "completed": 0, "total": 470})
    print("Loading existing full index and lexical statistics", flush=True)
    master = ReadOnlyIndex(
        Path(build["index_path"]), build["index_signature"], Path(build["source_directory"])
    )
    shared = ((master.frequencies, master.total, master.document_count), master.lexical_statistics)
    master.close()
    activity = {
        "lock": threading.Lock(),
        "http_active": 0,
        "http_started": 0,
        "peak_http": 0,
        "statuses": Counter(),
    }
    results = {
        q["question_id"]: json.loads((output / f"{q['question_id']}.json").read_text())
        for q in eligible
        if (output / f"{q['question_id']}.json").exists()
    }
    failures = {}
    started = time.perf_counter()

    def report(phase):
        categories = aggregate(questions, results, failures)
        with activity["lock"]:
            requests = {
                key: dict(value) if isinstance(value, Counter) else value
                for key, value in activity.items()
                if key != "lock"
            }
        summary = {
            "phase": phase,
            "completed": len(results),
            "total": len(eligible),
            "failed": len(failures),
            "failures": failures,
            "by_category": categories,
            "http": requests,
            "seconds_this_invocation": time.perf_counter() - started,
            "reported_cost_usd": sum(
                cost
                for result in results.values()
                for cost in result["response_costs_usd"]
                if cost is not None
            ),
            "actual_models": sorted(
                {batch["model"] for r in results.values() for batch in r["jev_usage"]["batches"]}
            ),
        }
        save(output / "summary.json", summary)
        save(output / "progress.json", {k: v for k, v in summary.items() if k != "by_category"})
        temporary = output / "metrics_by_category.tmp"
        with temporary.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "category",
                    "total",
                    "eligible",
                    "completed",
                    "failed",
                    "not_applicable",
                    *[f"{m}_Recall@{k}" for m in ("sag2", "bm25") for k in KS],
                ]
            )
            for kind, row in categories.items():
                writer.writerow(
                    [
                        kind,
                        *[
                            row[key]
                            for key in (
                                "total_questions",
                                "eligible_questions",
                                "completed_questions",
                                "failed_questions",
                                "not_applicable_questions",
                            )
                        ],
                        *[row["recall"][m][f"Recall@{k}"] for m in ("sag2", "bm25") for k in KS],
                    ]
                )
        temporary.replace(output / "metrics_by_category.csv")

    report("running")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_question, q, build, shared, api_key, activity, output): q
            for q in eligible
            if q["question_id"] not in results
        }
        while futures:
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                question = futures.pop(future)
                identifier = question["question_id"]
                try:
                    result = future.result()
                except (aiohttp.ClientError, TimeoutError, ValueError, OSError) as error:
                    failures[identifier] = {
                        "question_type": question["question_type"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    save(output / f"{identifier}.error.json", failures[identifier])
                    print(f"FAILED {identifier}: {type(error).__name__}: {error}", flush=True)
                else:
                    results[identifier] = result
                    save(output / f"{identifier}.json", result)
                    (output / f"{identifier}.error.json").unlink(missing_ok=True)
                    print(
                        f"DONE {len(results)}/470 {identifier} {question['question_type']}",
                        flush=True,
                    )
            report("running")
    report("complete" if not failures else "incomplete")
    if failures:
        raise RuntimeError(f"{len(failures)} questions failed; see saved error records")


if __name__ == "__main__":
    main()
