#!/usr/bin/env python3
"""Frozen paper controls. Raw records remain on US; successful calls are resumable."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import aiohttp
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sag2
from sag2_controls import (
    RecordedJev,
    FeedbackTerms,
    build_bm25_graph,
    compact_json,
    request_totals,
    run_protocol,
    save_json,
    rank_jev,
)

OUT = ROOT / "output" / "paper-controls"
DATA = ROOT / "dataset"
CAPS = ROOT / "results" / "controls" / "request_caps.json"
ENV_FILE = ROOT / ".env"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
ACTUAL_MODEL = "typesafe/jev-1.13-20260917"
CONCURRENCY = 20


def compact_record(row):
    fields = (
        "question_index",
        "dataset_sample_id",
        "problem_hops",
        "metrics",
        "candidate_judgments",
        "successful_scoring_requests",
        "stage_results",
        "logical_usage",
        "reused_responses",
        "requested_candidate_limit",
    )
    return {key: row[key] for key in fields if key in row}


def metrics(gold, candidates):
    docs = list(dict.fromkeys(c["full_doc"] for c in candidates))
    gold = set(gold)
    return {
        name: value
        for k in (1, 2, 5, 10)
        for name, value in (
            (f"Recall@{k}", len(gold.intersection(docs[:k])) / len(gold)),
            (f"AllEvidence@{k}", float(gold.issubset(docs[:k]))),
        )
    }


def final_rank(rows, index):
    ranked = rank_jev(rows)
    return sag2.rank_dpp(ranked[:10], index.diversity_idf) + ranked[10:]


class Client(RecordedJev):
    def __init__(self, *args, success_cap, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_cap = success_cap
        self.successes = set()

    async def request(self, payload):
        digest = hashlib.sha256(compact_json(payload).encode()).hexdigest()
        folder = self.directory / "requests" / digest
        response_path = folder / "success.json"
        if digest not in self.successes and len(self.successes) >= self.success_cap:
            raise ValueError("Successful request budget exhausted before sending")
        if response_path.exists():
            data = json.loads(response_path.read_text())
            self.check_model(data["model"])
            self.successes.add(digest)
            return data
        save_json(folder / "request.json", {"endpoint": self.endpoint, "payload": payload})
        shared = OUT / "response-cache" / f"{digest}.json"
        if shared.exists():
            saved = json.loads(shared.read_text())
            data = saved["response"]
            self.check_model(data["model"])
            save_json(
                folder / "cache-origin.json",
                {"successful_response": saved["origin"], "request_sha256": digest},
            )
            save_json(response_path, data)
            self.successes.add(digest)
            return data
        sequence = len(list(folder.glob("attempt-*.json")))
        retries, failures = 0, Counter()
        while True:
            sequence += 1
            path = folder / f"attempt-{sequence:04d}.json"
            record = {
                "started_at": datetime.now(UTC).isoformat(),
                "attempt": sequence,
                "request_sha256": digest,
                "candidate_count": len(payload["questions"]),
            }
            queued = time.perf_counter()
            async with self.gate:
                record["queue_seconds"] = time.perf_counter() - queued
                started = time.perf_counter()
                self.new_requests += 1
                save_json(path, record)
                try:
                    async with self.session.post(
                        self.endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    ) as response:
                        raw = await response.text()
                        record.update(
                            status=response.status,
                            response_text=raw,
                            elapsed_seconds=time.perf_counter() - started,
                            request_id=response.headers.get("x-request-id"),
                        )
                        save_json(path, record)
                        if response.status == 200:
                            data = json.loads(raw)
                            self.check_model(data["model"])
                            usage = data["usage"]
                            for field in ("input_tokens", "output_tokens", "cost"):
                                if type(usage[field]) not in (int, float) or usage[field] < 0:
                                    raise ValueError(f"Invalid API usage.{field}")
                            data.update(http_attempts=sequence, failed_attempts=dict(failures))
                            save_json(response_path, data)
                            save_json(
                                shared,
                                {"origin": str(response_path.relative_to(ROOT)), "response": data},
                            )
                            self.successes.add(digest)
                            return data
                        failures[f"http_{response.status}"] += 1
                        if response.status != 499:
                            if (
                                response.status not in (408, 429, 500, 502, 503, 504)
                                or retries >= 3
                            ):
                                response.raise_for_status()
                            retries += 1
                except (TimeoutError, aiohttp.ClientConnectionError) as error:
                    record.update(
                        error_type=type(error).__name__,
                        error=str(error),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    save_json(path, record)
                    failures["timeout" if isinstance(error, TimeoutError) else "connection"] += 1
                    if retries >= 3:
                        raise
                    retries += 1
                except asyncio.CancelledError:
                    record.update(
                        error_type="CancelledError", elapsed_seconds=time.perf_counter() - started
                    )
                    save_json(path, record)
                    raise
            await asyncio.sleep(min(30, 2 ** min(sequence - 1, 5)))

    def check_model(self, model):
        if model != ACTUAL_MODEL:
            raise ValueError(f"Model identity changed: {model}")
        super().check_model(model)

    def pack(self, query, candidates):
        state = compact_json({"original_question": query, "previously_retrieved_evidence": []})
        state_tokens = len(self.encoding.encode(state, disallowed_special=())) + 1024
        batches, batch, total = [], [], state_tokens
        for c in candidates:
            question = {
                "type": self.question_type,
                "instructions": {
                    "question": sag2.CRITERION,
                    "candidate_title": c["title"],
                    "candidate_passage": c["content"],
                },
                "criteria": {
                    "true": "This candidate itself supplies a necessary fact in a supported answer chain, with the required entity identities and relationships.",
                    "false": "The candidate is merely related background, refers to the wrong entity or relationship, or the useful fact occurs only in other context passages.",
                },
            }
            tokens = len(self.encoding.encode(compact_json(question), disallowed_special=())) + 64
            if state_tokens + tokens > 32000:
                raise ValueError("Candidate exceeds the untruncated context limit")
            if total + tokens > 64000:
                batches.append(batch)
                batch, total = [], state_tokens
            batch.append(c)
            total += tokens
        if batch:
            batches.append(batch)
        if len(batches) > self.success_cap:
            raise ValueError(
                f"Matched budget needs {len(batches)} batches; success cap {self.success_cap}"
            )
        return batches


def records():
    samples = json.loads((DATA / "musique.json").read_text())
    rows = []
    for i, q in enumerate(samples):
        gold = list(
            dict.fromkeys(
                f"{p['title']}\n{p['paragraph_text']}"
                for p in q["paragraphs"]
                if p["is_supporting"]
            )
        )
        rows.append(
            {
                "question_index": i,
                "dataset_sample_id": q["id"],
                "question": q["question"],
                "gold_docs": gold,
                "problem_hops": int(q["id"][0]),
            }
        )
    assert len(rows) == 1000 and len({r["dataset_sample_id"] for r in rows}) == 1000
    return rows


def summarize(destination, rows):
    summary = {
        "status": "completed",
        "question_count": len(rows),
        "candidate_judgments": sum(r["candidate_judgments"] for r in rows),
        "successful_scoring_requests": sum(r["successful_scoring_requests"] for r in rows),
        "request_totals": request_totals(destination),
        "groups": {},
    }
    summary["logical_usage_including_reused_responses"] = {
        key: sum(r["logical_usage"][key] for r in rows)
        for key in ("input_tokens", "output_tokens", "reported_cost_usd")
    }
    summary["reused_responses"] = sum(r["reused_responses"] for r in rows)
    for group in ("all", "2", "3", "4"):
        selected = rows if group == "all" else [r for r in rows if str(r["problem_hops"]) == group]
        summary["groups"][group] = {
            "n": len(selected),
            "metrics": {
                k: sum(r["metrics"][k] for r in selected) / len(selected)
                for k in selected[0]["metrics"]
            },
        }
    summary["stages"] = {}
    for name in rows[0]["stage_results"]:
        selected = [r["stage_results"][name] for r in rows if name in r["stage_results"]]
        summary["stages"][name] = {
            "n": len(selected),
            "metrics": {
                k: sum(r["metrics"][k] for r in selected) / len(selected)
                for k in selected[0]["metrics"]
            },
        }
    if "requested_candidate_limit" in rows[0]:
        summary["budget_matching"] = {
            "requested_judgments": sum(r["requested_candidate_limit"] for r in rows),
            "actual_judgments": summary["candidate_judgments"],
        }
    save_json(destination / "summary.json", summary)
    with (destination / "details.jsonl").open("w") as stream:
        for r in rows:
            stream.write(
                (
                    destination / "questions" / f"{r['question_index']:06d}" / "result.json"
                ).read_text()
                + "\n"
            )
    return summary


async def variant(name, mode, index, data, caps, session, gate, state, feedback=None, graph=None):
    destination = OUT / name
    completed = {
        int(p.parent.name): compact_record(json.loads(p.read_text()))
        for p in destination.glob("questions/*/result.json")
    }
    pending = iter(r for r in data if r["question_index"] not in completed)
    state.update(
        active_variant=name, completed_questions=len(completed), selected_questions=len(data)
    )
    save_json(OUT / "status.json", state)
    reference = OUT / "sag2_h2" / "questions"

    async def worker():
        for r in pending:
            i = r["question_index"]
            folder = destination / "questions" / f"{i:06d}"
            client = Client(
                session,
                os.environ["JEV_API_KEY"],
                folder,
                gate,
                OUT / "actual-model.json",
                no_context=mode == "no_context",
                success_cap=caps[r["dataset_sample_id"]],
            )
            save_json(folder / "started.json", {"started_at": datetime.now(UTC).isoformat()})
            if mode == "matched":
                ref = json.loads((reference / f"{i:06d}" / "result.json").read_text())
                k = ref["candidate_judgments"]
                hits = index.retrieve(index.match_query(r["question"]), set(), k)
                pool = [{**h, "fusion_score": 1 / (61 + j)} for j, h in enumerate(hits)]
                scored = []
                for batch in client.pack(r["question"], pool):
                    output, _ = await client.score(r["question"], batch, [])
                    scored.extend(output)
                result = {
                    "candidates": final_rank(scored, index),
                    "trace": [],
                    "requested_candidate_limit": k,
                }
            else:
                result = await run_protocol(
                    index, client, r["question"], mode, feedback=feedback, graph=graph
                )
                if mode not in ("sag2", "no_context", "rm3", "gar"):
                    result["candidates"] = final_rank(result["candidates"], index)
            candidates = result.pop("candidates")
            stages = {}
            if "baseline" in result:
                stages["bm25"] = {
                    "metrics": metrics(r["gold_docs"], result["baseline"]),
                    "candidate_judgments": 0,
                }
            for j, s in enumerate(client.stages):
                if mode in ("sag2", "no_context"):
                    ranked = final_rank(s["scored_candidates"], index)
                    stages[f"h{j}"] = {
                        "metrics": metrics(r["gold_docs"], ranked),
                        "candidate_judgments": sum(
                            x["usage"]["judgments"] for x in client.stages[: j + 1]
                        ),
                    }
            row = {
                **r,
                **result,
                "final_candidates": candidates,
                "retrieved_docs": list(dict.fromkeys(c["full_doc"] for c in candidates)),
                "metrics": metrics(r["gold_docs"], candidates),
                "stage_results": stages,
                "candidate_judgments": sum(s["usage"]["judgments"] for s in client.stages),
                "successful_scoring_requests": len(client.successes),
                "successful_request_cap": client.success_cap,
            }
            responses = [
                json.loads((folder / "requests" / digest / "success.json").read_text())
                for digest in client.successes
            ]
            row["logical_usage"] = {
                "input_tokens": sum(x["usage"]["input_tokens"] for x in responses),
                "output_tokens": sum(x["usage"]["output_tokens"] for x in responses),
                "reported_cost_usd": sum(x["usage"]["cost"] for x in responses),
            }
            row["reused_responses"] = len(list(folder.glob("requests/*/cache-origin.json")))
            save_json(folder / "result.json", row)
            completed[i] = compact_record(row)
            state.update(
                completed_questions=len(completed), updated_at=datetime.now(UTC).isoformat()
            )
            save_json(OUT / "status.json", state)
            if len(completed) % 10 == 0:
                print(name, len(completed), "/", len(data), flush=True)

    async with asyncio.TaskGroup() as group:
        for _ in range(CONCURRENCY):
            group.create_task(worker())
    rows = [completed[r["question_index"]] for r in data]
    summary = summarize(destination, rows)
    state["variants"][name] = summary
    save_json(OUT / "status.json", state)
    return rows


async def main():
    load_dotenv(ENV_FILE)
    if os.getenv("OPENROUTER_API_KEY"):
        os.environ["JEV_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    if not os.getenv("JEV_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY or JEV_API_KEY")
    data = records()
    caps = json.loads(CAPS.read_text())
    assert set(caps) == {r["dataset_sample_id"] for r in data}
    corpus = json.loads((DATA / "musique_corpus.json").read_text())
    signature = sag2.corpus_fingerprint(DATA / "musique_corpus.json")
    identity = {
        "sources": {
            name: sag2.corpus_fingerprint(ROOT / "scripts" / name)
            for name in ("run_paper_controls.py", "sag2.py", "sag2_controls.py", "chunking.py")
        },
        "questions_sha256": sag2.corpus_fingerprint(DATA / "musique.json"),
        "corpus_sha256": signature,
        "endpoint": ENDPOINT,
        "requested_model": MODEL,
        "expected_model": ACTUAL_MODEL,
        "concurrency": CONCURRENCY,
        "jev_question_type": "noul",
        "chunk_chars": sag2.DEFAULT_CHUNK_CHARS,
        "chunk_policy": sag2.CHUNK_POLICY,
        "index_workers": 8,
        "final_candidate_k": 100,
        "common_postprocessing": "DPP top10 for all JEV methods; none for lexical-only methods",
        "matched_budget": "per-question current reference judgments; one fixed BM25 pool; empty context; token-packed without truncation",
        "successful_request_caps": caps,
        "request_caps_sha256": sag2.corpus_fingerprint(CAPS),
    }
    manifest = OUT / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity, (
            "Source or protocol changed; resume refused"
        )
    else:
        save_json(manifest, identity)
    state = {
        "status": "running",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "endpoint": ENDPOINT,
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "started_at": datetime.now(UTC).isoformat(),
        "variants": {},
    }
    save_json(OUT / "status.json", state)
    try:
        start = time.perf_counter()
        index = sag2.CorpusIndex(
            OUT / "cache" / "corpus.sqlite", corpus, signature, sag2.DEFAULT_CHUNK_CHARS, index_workers=8
        )
        if not (OUT / "index-cost.json").exists():
            save_json(
                OUT / "index-cost.json",
                {
                    "cold_construction_seconds": time.perf_counter() - start,
                    "source_records": len(corpus),
                    "indexed_units": index.total,
                    "sqlite_bytes": (OUT / "cache" / "corpus.sqlite").stat().st_size,
                    "offline_model_requests": 0,
                    "offline_model_cost_usd": 0,
                    "index_workers": 8,
                    "host": socket.gethostname(),
                },
            )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            gate = asyncio.Semaphore(CONCURRENCY)
            ref = await variant("sag2_h2", "sag2", index, data, caps, session, gate, state)
            cost_rows = ref[:100]
            usage = {"input_tokens": 0, "output_tokens": 0, "reported_cost_usd": 0.0}
            for r in cost_rows:
                folder = OUT / "sag2_h2" / "questions" / f"{r['question_index']:06d}"
                for p in folder.glob("requests/*/attempt-*.json"):
                    a = json.loads(p.read_text())
                    if "response_text" in a:
                        try:
                            response = json.loads(a["response_text"])
                        except json.JSONDecodeError:
                            continue
                        u = response.get("usage")
                        if u:
                            usage["input_tokens"] += u["input_tokens"]
                            usage["output_tokens"] += u["output_tokens"]
                            usage["reported_cost_usd"] += u["cost"]
            save_json(
                OUT / "cost-100.json",
                {
                    "n": 100,
                    "question_indices": list(range(100)),
                    "source": "first100_current_full_reference_measured_requests",
                    "usage": usage,
                    "candidate_judgments": sum(r["candidate_judgments"] for r in cost_rows),
                    "successful_scoring_requests": sum(
                        r["successful_scoring_requests"] for r in cost_rows
                    ),
                    "index": json.loads((OUT / "index-cost.json").read_text()),
                    "online_latency": "not reported; provider queueing is not separable",
                },
            )
            for name, mode in [
                ("bm25_jev_matched_budget", "matched"),
                ("sag2_no_context", "no_context"),
                ("lexical_final_jev", "final_only"),
            ]:
                await variant(name, mode, index, data, caps, session, gate, state)
            feedback = FeedbackTerms(index)
            for name, mode in [("rm3", "rm3"), ("rm3_final_jev", "rm3_jev")]:
                await variant(
                    name, mode, index, data, caps, session, gate, state, feedback=feedback
                )
            state.update(active_variant="gar_graph", updated_at=datetime.now(UTC).isoformat())
            save_json(OUT / "status.json", state)
            graph, timing = build_bm25_graph(index, OUT / "cache" / "gar.sqlite", signature)
            save_json(OUT / "gar-graph-cost.json", timing)
            for name, mode in [("gar_bm25", "gar"), ("gar_bm25_final_jev", "gar_jev")]:
                await variant(name, mode, index, data, caps, session, gate, state, graph=graph)
        state.update(
            status="completed", active_variant=None, finished_at=datetime.now(UTC).isoformat()
        )
        save_json(OUT / "status.json", state)
    except BaseException as error:
        state.update(
            status="needs_attention",
            error=f"{type(error).__name__}: {error}",
            updated_at=datetime.now(UTC).isoformat(),
        )
        save_json(OUT / "status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the eight MuSiQue paper controls")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dataset-dir", type=Path, default=DATA)
    parser.add_argument("--request-caps", type=Path, default=CAPS)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 20:
        parser.error("concurrency must be between 1 and 20")
    OUT, DATA, CAPS, ENV_FILE = args.output_dir, args.dataset_dir, args.request_caps, args.env_file
    CONCURRENCY = args.concurrency
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main())
