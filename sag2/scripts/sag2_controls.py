"""Retrieval protocols and durable model records for the MuSiQue experiments."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from sag2 import (
    CorpusIndex,
    JevClient,
    compact_json,
    quote_term,
    search_sag2,
    tokenize,
)

GRAPH_SCHEMA = 1
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODEL = "typesafe/jev-1.13"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RecordedJev(JevClient):
    """One question's calls, with a suite-wide HTTP concurrency limit and request cache."""

    def __init__(self, session, api_key, directory, gate, model_identity, *, no_context=False):
        super().__init__(session, OPENROUTER_ENDPOINT, OPENROUTER_MODEL, api_key)
        self.directory = Path(directory)
        self.gate = gate
        self.model_identity = model_identity
        self.no_context = no_context
        self.stages = []
        self.new_requests = 0

    def check_model(self, model):
        if self.model_identity.exists():
            expected = json.loads(self.model_identity.read_text())["actual_model"]
            if expected != model:
                raise ValueError(f"JEV model changed: expected {expected}, received {model}")
        else:
            save_json(self.model_identity, {"actual_model": model})

    async def score(self, query, candidates, evidence):
        evidence = [] if self.no_context else evidence
        scored, usage = await super().score(query, candidates, evidence)
        stage = {
            "round": len(self.stages),
            "original_question": query,
            "candidates": candidates,
            "evidence": evidence,
            "scored_candidates": scored,
            "usage": usage,
        }
        save_json(self.directory / "scores" / f"round-{len(self.stages):02d}.json", stage)
        self.stages.append(stage)
        return scored, usage


def request_totals(directory: Path) -> dict:
    totals = {
        "http_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reported_cost_usd": 0.0,
        "attempts_without_usage": 0,
    }
    for path in directory.glob("questions/*/requests/*/attempt-*.json"):
        row = json.loads(path.read_text())
        totals["http_requests"] += 1
        if "response_text" in row:
            try:
                response = json.loads(row["response_text"])
            except json.JSONDecodeError:
                response = {}  # An error response may be HTML; its complete text is saved.
            usage = response.get("usage")
            if usage is not None:
                totals["input_tokens"] += usage["input_tokens"]
                totals["output_tokens"] += usage["output_tokens"]
                totals["reported_cost_usd"] += usage["cost"]
            else:
                totals["attempts_without_usage"] += 1
        else:
            totals["attempts_without_usage"] += 1
        totals["successful_requests" if row.get("status") == 200 else "failed_requests"] += 1
    totals["billing_complete"] = totals["attempts_without_usage"] == 0
    return totals


def rank_jev(candidates):
    return sorted(candidates, key=lambda row: (-row["jev_score"], -row["fusion_score"], row["id"]))


async def lexical_expansion(
    index: CorpusIndex,
    client,
    query: str,
    *,
    candidate_k=100,
    seed_k=10,
    max_hops=2,
    term_limit=20,
    final_k=100,
):
    """Lexical seeds and uniform weights; retain prior seeds and final-round candidates."""
    expression = index.match_query(query)
    seen, frontier, trace = set(), [], []
    protected, carried, pool = {}, {}, []
    for hop in range(max_hops + 1):
        terms = (
            index.expansion_terms(
                query, [{**row, "jev_score": 1.0} for row in frontier], term_limit
            )
            if hop
            else []
        )
        if not expression or (hop and not terms):
            break
        hits, queries = (
            index.retrieve_expansion(expression, terms, seen, candidate_k)
            if hop
            else (index.retrieve(expression, seen, candidate_k), [])
        )
        if not hits:
            break
        for i, hit in enumerate(hits):
            hit.update(hop=hop, fusion_score=1 / (61 + i))
        if hop == 0:
            protected = {hit["id"]: hit for hit in hits[:5]}
        carried.update({row["id"]: row for row in frontier})
        pool = list(
            {row["id"]: row for row in [*protected.values(), *carried.values(), *hits]}.values()
        )
        seen.update(hit["id"] for hit in hits)
        frontier = hits[:seed_k]
        trace.append(
            {
                "stage": "lexical_expansion",
                "hop": hop,
                "candidate_ids": [row["id"] for row in hits],
                "seed_ids": [row["id"] for row in frontier],
                "term_scores": [{**term, "seed_score_source": "uniform"} for term in terms],
                "expansion_queries": queries,
            }
        )
    candidates = pool[:final_k]
    final_ids = [row["id"] for row in candidates]
    if client is not None and candidates:
        scored, _ = await client.score(query, candidates, [])
        candidates = rank_jev(scored)
    return {
        "candidates": candidates,
        "trace": trace,
        "final_scoring_context": "none",
        "final_pool_policy": "protected_BM25_then_carried_seeds_then_final_round_candidates",
        "final_pool_before_limit_ids": [row["id"] for row in pool],
        "final_scoring_candidate_ids": final_ids,
    }


class FeedbackTerms:
    """FTS5 body occurrence counts shared by all RM3 queries."""

    def __init__(self, index: CorpusIndex):
        self.index = index
        index.db.execute(
            "CREATE VIRTUAL TABLE temp.rm3_instances USING fts5vocab(main,chunk_fts,'instance')"
        )
        self.counts = defaultdict(Counter)
        for doc, term in index.db.execute(
            "SELECT doc,term FROM temp.rm3_instances WHERE col='body'"
        ):
            self.counts[doc][term] += 1
        index.db.execute("CREATE VIRTUAL TABLE temp.rm3_query USING fts5(body)")
        index.db.execute(
            "CREATE VIRTUAL TABLE temp.rm3_query_instances USING fts5vocab(temp,rm3_query,'instance')"
        )

    def expand(self, query: str, *, fb_docs=10, fb_terms=10, query_weight=0.5):
        initial = self.index.retrieve(self.index.match_query(query), set(), fb_docs)
        feedback, document_terms = defaultdict(float), []
        for document in initial:
            valid = {
                term: tf
                for term, tf in self.counts[document["id"]].items()
                if 2 <= len(term) <= 20
                and re.fullmatch(r"[a-z0-9]+", term)
                and self.index.frequency(term) / self.index.total <= 0.1
            }
            kept = sorted(valid.items(), key=lambda pair: (-pair[1], pair[0]))[:fb_terms]
            norm = sum(tf for _, tf in kept)
            document_terms.append(
                {
                    "id": document["id"],
                    "positive_bm25": -document["score"],
                    "retained_terms": kept,
                    "normalizer": norm,
                }
            )
            for term, tf in kept:
                feedback[term] += -document["score"] * tf / norm
        kept_feedback = sorted(feedback.items(), key=lambda pair: (-pair[1], pair[0]))[:fb_terms]
        norm = sum(weight for _, weight in kept_feedback)
        feedback_model = {term: weight / norm for term, weight in kept_feedback}
        db = self.index.db
        db.execute("DELETE FROM temp.rm3_query")
        db.execute(
            "INSERT INTO temp.rm3_query(rowid,body) VALUES(1,?)", (" ".join(tokenize(query)),)
        )
        query_counts = Counter(
            row[0] for row in db.execute("SELECT term FROM temp.rm3_query_instances")
        )
        query_length = sum(query_counts.values())
        weights = {term: query_weight * tf / query_length for term, tf in query_counts.items()}
        for term, weight in feedback_model.items():
            weights[term] = weights.get(term, 0) + (1 - query_weight) * weight
        return weights, {
            "feedback_documents": document_terms,
            "feedback_model": feedback_model,
            "query_counts": dict(query_counts),
            "query_weight": query_weight,
            "weighted_query": weights,
        }


def weighted_bm25(index: CorpusIndex, weights: dict[str, float], limit: int):
    scores = defaultdict(float)
    for term, weight in weights.items():
        for identifier, score in index.db.execute(
            "SELECT rowid,bm25(chunk_fts,2,2,1) FROM chunk_fts WHERE chunk_fts MATCH ?",
            (quote_term(term),),
        ):
            scores[identifier] += weight * -score
    order = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit]
    rows = {
        row["id"]: dict(row)
        for row in index.db.execute(
            "SELECT * FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
            (compact_json(order),),
        )
    }
    return [
        {**rows[identifier], "score": -scores[identifier], "fusion_score": 1 / (61 + i)}
        for i, identifier in enumerate(order)
    ]


async def rm3_search(index, feedback, client, query, *, candidate_k=100):
    weights, trace = feedback.expand(query)
    candidates = weighted_bm25(index, weights, candidate_k)
    trace["candidate_ids"] = [row["id"] for row in candidates]
    if client is not None and candidates:
        scored, _ = await client.score(query, candidates, [])
        candidates = rank_jev(scored)
    return {
        "candidates": candidates,
        "trace": [{"stage": "rm3", **trace}],
        "final_scoring_context": "none",
    }


def build_bm25_graph(index: CorpusIndex, path: Path, fingerprint: str, *, neighbors=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = sqlite3.connect(path)
    signature = compact_json(
        [
            GRAPH_SCHEMA,
            fingerprint,
            neighbors,
            "full_body_query_no_80_word_cap",
            "bm25_columns_2_2_1",
        ]
    )
    graph.execute("CREATE TABLE IF NOT EXISTS metadata(signature TEXT NOT NULL)")
    saved = graph.execute("SELECT signature FROM metadata").fetchone()
    if saved is not None and saved[0] != signature:
        raise ValueError("GAR graph cache belongs to a different corpus or protocol")
    if saved is None:
        graph.execute("INSERT INTO metadata VALUES(?)", (signature,))
    graph.execute(
        "CREATE TABLE IF NOT EXISTS neighbors(source INTEGER PRIMARY KEY, entries TEXT NOT NULL)"
    )
    started = time.perf_counter()
    completed = {row[0] for row in graph.execute("SELECT source FROM neighbors")}
    added = 0
    for document in index.db.execute("SELECT id,content FROM chunks ORDER BY id").fetchall():
        if document["id"] in completed:
            continue
        expression = " OR ".join(
            quote_term(term) for term in dict.fromkeys(tokenize(document["content"]))
        )
        hits = index.retrieve(expression, {document["id"]}, neighbors) if expression else []
        graph.execute(
            "INSERT INTO neighbors VALUES(?,?)",
            (
                document["id"],
                compact_json([{"id": hit["id"], "bm25_score": hit["score"]} for hit in hits]),
            ),
        )
        added += 1
        if added % 100 == 0:
            graph.commit()
            print(f"GAR 图：已完成 {len(completed) + added}/{index.total}", flush=True)
    graph.commit()
    entries = {
        row[0]: json.loads(row[1]) for row in graph.execute("SELECT source,entries FROM neighbors")
    }
    graph.close()
    return entries, {
        "source_count": len(entries),
        "new_source_queries": added,
        "seconds_this_invocation": time.perf_counter() - started,
        "graph_bytes": path.stat().st_size,
        "neighbors": neighbors,
        "signature": signature,
        "path": str(path),
    }


async def gar_search(
    index,
    graph,
    client,
    query,
    *,
    candidate_k=100,
    batch_size=16,
    initial_k=1000,
):
    """GAR with BM25 feedback and optional final JEV."""
    expression = index.match_query(query)
    initial = index.retrieve(expression, set(), initial_k) if expression else []
    initial_queue = deque(row["id"] for row in initial)
    bm25 = (
        {
            identifier: -score
            for identifier, score in index.db.execute(
                "SELECT rowid,bm25(chunk_fts,2,2,1) FROM chunk_fts WHERE chunk_fts MATCH ?",
                (expression,),
            )
        }
        if expression
        else {}
    )
    frontier, scored, trace, sequence, turn = {}, {}, [], 0, 0
    while len(scored) < candidate_k and (initial_queue or frontier):
        from_initial = turn % 2 == 0
        turn += 1
        take = min(batch_size, candidate_k - len(scored))
        batch = []
        if from_initial:
            while initial_queue and len(batch) < take:
                identifier = initial_queue.popleft()
                if identifier not in scored:
                    batch.append(identifier)
        else:
            batch = sorted(
                frontier,
                key=lambda identifier: (
                    -frontier[identifier][0],
                    frontier[identifier][1],
                    identifier,
                ),
            )[:take]
        if not batch:
            continue
        parents = {}
        for identifier in batch:
            if identifier in frontier:
                parents[identifier] = frontier[identifier][2]
            scored[identifier] = bm25.get(
                identifier, 0.0
            )  # No query terms means BM25 contribution zero.
            frontier.pop(identifier, None)
        initial_queue = deque(
            identifier for identifier in initial_queue if identifier not in scored
        )
        for parent in sorted(batch, key=lambda identifier: (-scored[identifier], identifier)):
            for edge in graph[parent]:
                identifier = edge["id"]
                if identifier in scored:
                    continue
                priority = scored[parent]
                if identifier not in frontier:
                    frontier[identifier] = (priority, sequence, parent)
                    sequence += 1
                elif priority > frontier[identifier][0]:
                    frontier[identifier] = (priority, frontier[identifier][1], parent)
        trace.append(
            {
                "stage": "gar_bm25",
                "source": "initial" if from_initial else "graph",
                "candidate_ids": batch,
                "parent_ids": parents,
                "evidence_ids": [],
                "scores": {identifier: scored[identifier] for identifier in batch},
                "frontier_ids": list(frontier),
            }
        )
    rows = {
        row["id"]: dict(row)
        for row in index.db.execute(
            "SELECT * FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
            (compact_json(list(scored)),),
        )
    }
    order = sorted(scored, key=lambda identifier: (-scored[identifier], identifier))
    candidates = [
        {**rows[identifier], "score": -bm25.get(identifier, 0.0), "fusion_score": 1 / (61 + i)}
        for i, identifier in enumerate(order)
    ]
    if client is not None and candidates:
        output, _ = await client.score(query, candidates, [])
        candidates = rank_jev(output)
    return {
        "candidates": candidates,
        "trace": trace,
        "final_scoring_context": "none",
        "gar_expansion_scorer": "original_query_BM25",
    }


async def run_protocol(index, client, query, mode, *, candidate_k=100, feedback=None, graph=None):
    if mode in ("sag2", "no_context"):
        result = await search_sag2(
            index, client, query, candidate_k=100, seed_k=10, max_hops=2, expansion_term_limit=20
        )
        if mode == "no_context":
            for stage in result["trace"]:
                if stage["stage"] == "jev":
                    stage["evidence_ids"] = []
        return result
    if mode == "final_only":
        return await lexical_expansion(index, client, query, final_k=candidate_k)
    if mode in ("rm3", "rm3_jev"):
        return await rm3_search(
            index,
            feedback,
            client if mode.endswith("_jev") else None,
            query,
            candidate_k=candidate_k,
        )
    if mode in ("gar", "gar_jev"):
        return await gar_search(
            index,
            graph,
            client if mode.endswith("_jev") else None,
            query,
            candidate_k=candidate_k,
        )
    raise ValueError(f"Unknown experiment mode: {mode}")
