import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from scripts import run_sag2_benchmark as benchmark

sag2 = __import__("sag2")
ORACLE = json.loads(Path(__file__).with_name("sag2_reference.json").read_text())


def test_corpus_precheck_preserves_source_whitespace(tmp_path, caplog):
    body = "   This passage begins with spaces."
    (tmp_path / "hotpotqa_corpus.json").write_text(json.dumps([
        {"title": "Town", "text": body}
    ]))
    (tmp_path / "hotpotqa.json").write_text(json.dumps([
        {"_id": "q1", "question": "Where?", "supporting_facts": [["Town", 0]],
         "context": [["Town", [body]]]}
    ]))
    _, records, _ = benchmark.load_dataset("hotpotqa", tmp_path)
    assert records[0]["gold_docs"] == [f"Town\n{body}"]
    assert "do not exactly match the corpus" not in caplog.text


class OracleJev(sag2.JevClient):
    """Fixed judgments used by the independent frozen SAG2 reference run."""

    def __init__(self):
        super().__init__(None, "http://localhost/decisions", "jev-latest", "local-test")
        self.payloads = []

    async def request(self, payload):
        self.payloads.append(payload)
        state = json.loads(payload["state"])
        evidence = [row["title"] for row in state["previously_retrieved_evidence"]]
        answers = {}
        for key, question in payload["questions"].items():
            title = question["instructions"]["candidate_title"]
            score = 0.01
            if title == "Anchor":
                score = 0.8
            elif title == "Multiple":
                score = 0.95 if "Anchor" in evidence else 0.1
            elif title == "Bridge":
                score = 0.99 if "Multiple" in evidence else 0.05
            elif title == "High":
                score = 0.7
            elif title == "Series":
                score = 0.08
            answers[key] = {"type": "noul", "noul": score}
        return {
            "model": "oracle-jev",
            "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "request_attempts": 1,
        }


@pytest.mark.parametrize("row", ORACLE["tokenizer"])
def test_tokenizer_matches_frozen_sag2(row):
    assert sag2.tokenize(row["text"]) == row["tokens"]


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["query"])
@pytest.mark.parametrize("index_workers", [1, 4])
def test_rankings_expansion_and_entire_scoring_payload_match_frozen_sag2(
    tmp_path, case, index_workers
):
    index = sag2.CorpusIndex(
        tmp_path / "oracle.sqlite", ORACLE["corpus"], "fixture", index_workers=index_workers
    )
    client = OracleJev()
    try:
        result = asyncio.run(sag2.search_sag2(index, client, case["query"], **case["options"]))
        assert [row["id"] for row in result["candidates"]] == case["result"]["ranked_ids"]
        assert [row["jev_score"] for row in result["candidates"]] == case["result"]["ranked_scores"]
        rounds = [row for row in result["trace"] if row["stage"] in ("bm25", "expansion")]
        assert len(rounds) == len(case["result"]["rounds"])
        for actual, expected in zip(rounds, case["result"]["rounds"]):
            assert actual["candidate_document_ids"] == expected["ids"]
            assert actual["terms"] == expected["terms"]
            for term, old_term in zip(actual["term_scores"], expected["term_scores"]):
                for key, value in old_term.items():
                    if key == "seed_document_id":
                        assert term[key] == int(value)
                    elif isinstance(value, (float, int)):
                        assert term[key] == pytest.approx(value)
                    else:
                        assert term[key] == value
            assert [q["ranking_expression"] for q in actual["expansion_queries"]] == [
                q["ranking_expression"] for q in expected["queries"]
            ]
        assert client.payloads == case["requests"]
        assert result["jev_usage"]["judgments"] == case["result"]["judgments"]
    finally:
        index.close()


def test_all_expansion_terms_contribute_before_candidate_limit(tmp_path):
    index = sag2.CorpusIndex(tmp_path / "multiple.sqlite", ORACLE["corpus"], "fixture")
    try:
        hits, _ = index.retrieve_expansion(
            '"birthplace"',
            [
                {"term": "alice paul", "kind": "name_phrase"},
                {"term": "bob smith", "kind": "name_phrase"},
            ],
            {1},
            1,
        )
        assert hits[0]["title"] == "Multiple"
    finally:
        index.close()


@pytest.mark.parametrize("term", [None, "bridge"])
@pytest.mark.parametrize("limit", [1, 7, 100])
def test_candidate_limit_preserves_scores_exclusions_and_ties(tmp_path, term, limit):
    corpus = [
        {"title": "Document", "text": "target " * (i // 3 + 1) + ("bridge" if i % 2 else "filler")}
        for i in range(240)
    ]
    index = sag2.CorpusIndex(tmp_path / "limit.sqlite", corpus, "limit", chunk_chars=10000)
    try:
        excluded = {1, 2, 3, 232, 233, 234}
        sql = (
            "SELECT c.*,bm25(chunk_fts,2,2,1) AS score FROM chunk_fts "
            "JOIN chunks c ON c.id=chunk_fts.rowid WHERE chunk_fts MATCH ? "
            "AND c.id NOT IN (SELECT value FROM json_each(?)) "
        )
        params = ['"target" OR "bridge"', sag2.compact_json(sorted(excluded))]
        if term is not None:
            sql += "AND c.id IN (SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?) "
            params.append(sag2.quote_term(term))
        expected = [
            dict(row)
            for row in index.db.execute(sql + "ORDER BY score,c.id LIMIT ?", [*params, limit])
        ]
        assert index.retrieve(params[0], excluded, limit, term) == expected
    finally:
        index.close()


def test_full_text_is_read_only_after_candidate_limit(tmp_path):
    corpus = [
        {"title": "Document", "text": "target " * (i + 1) + "filler " * (240 - i)}
        for i in range(240)
    ]
    index = sag2.CorpusIndex(tmp_path / "late-read.sqlite", corpus, "late-read", chunk_chars=10000)
    reads = []

    def read_full_doc(text):
        reads.append(text)
        return text

    try:
        index.db.create_function("read_full_doc", 1, read_full_doc)
        index.db.execute("ALTER TABLE chunks RENAME TO stored_chunks")
        index.db.execute(
            "CREATE VIEW chunks AS SELECT id,document_id,title,content,"
            "read_full_doc(full_doc) AS full_doc,source_id FROM stored_chunks"
        )
        actual = index.retrieve('"target"', set(), 100)
        assert len(actual) == 100
        assert len(reads) == 100
    finally:
        index.close()


def test_query_cache_preserves_filters_limit_and_only_stores_ids_and_scores(tmp_path):
    index = sag2.CorpusIndex(tmp_path / "query-cache.sqlite", ORACLE["corpus"], "fixture")
    try:
        expression = '"anchor" OR "bridge"'
        original = index.retrieve(expression, set(), 100)
        assert index.retrieve(expression, set(), 100) == original
        assert index._query_cache_hits == 1
        assert all(
            isinstance(identifier, int) and isinstance(score, float)
            for rows in index._query_cache.values()
            for identifier, score in rows
        )
        original[0]["title"] = "mutated caller result"
        assert index.retrieve(expression, set(), 100)[0]["title"] != "mutated caller result"
        excluded = {original[0]["id"]}
        assert all(row["id"] not in excluded for row in index.retrieve(expression, excluded, 100))
        assert len(index.retrieve(expression, set(), 1)) == 1
        required = index.retrieve(expression, set(), 100, "bridge")
        assert all("bridge" in (row["title"] + row["content"]).lower() for row in required)
        assert index._query_cache_misses == 4
    finally:
        index.close()


def test_query_cache_invalidates_own_writes_external_writes_and_keeps_snapshot(tmp_path):
    path = tmp_path / "query-invalidation.sqlite"
    corpus = [{"title": f"Doc {i}", "text": f"anchor evidence {i}"} for i in range(4)]
    index = sag2.CorpusIndex(path, corpus, "fixture")
    try:
        index.db.execute("PRAGMA journal_mode=WAL")
        original = index.retrieve('"anchor"', set(), 100)
        with index.db:
            index.db.execute("DELETE FROM chunk_fts WHERE rowid=1")
            index.db.execute("DELETE FROM chunks WHERE id=1")
        assert len(index.retrieve('"anchor"', set(), 100)) == len(original) - 1
        index.db.execute("BEGIN")
        index.db.execute("SELECT * FROM metadata").fetchall()
        with sqlite3.connect(path) as writer:
            writer.execute("DELETE FROM chunk_fts WHERE rowid=2")
            writer.execute("DELETE FROM chunks WHERE id=2")
        # A live read transaction must retain its old snapshot, including cache hits.
        assert any(row["id"] == 2 for row in index.retrieve('"anchor"', set(), 100))
        index.db.rollback()
        assert all(row["id"] not in {1, 2} for row in index.retrieve('"anchor"', set(), 100))
        assert index._query_cache_misses == 3
    finally:
        index.close()


def test_query_cache_evicts_least_recently_used_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(sag2, "QUERY_CACHE_ENTRIES", 2)
    index = sag2.CorpusIndex(tmp_path / "lru.sqlite", ORACLE["corpus"], "fixture")
    try:
        for limit in (1, 2, 1, 3):
            index.retrieve('"anchor"', set(), limit)
        assert [key[2] for key in index._query_cache] == [1, 3]
        index.retrieve('"anchor"', set(), 2)
        assert index._query_cache_misses == 4
    finally:
        index.close()


def test_parallel_expansion_matches_serial_caches_and_refreshes_after_update(tmp_path, monkeypatch):
    path = tmp_path / "parallel-queries.sqlite"
    corpus = [{"title": f"Doc {i}", "text": "alpha beta gamma " + "filler " * i} for i in range(12)]
    index = sag2.CorpusIndex(path, corpus, "fixture")
    terms = [{"term": term, "kind": "word"} for term in ("alpha", "beta", "gamma")]
    try:
        monkeypatch.setattr(sag2, "QUERY_WORKERS", 1)
        expected = index.retrieve_expansion('"alpha"', terms, {2}, 10)
        monkeypatch.setattr(sag2, "QUERY_WORKERS", 8)
        assert index.retrieve_expansion('"alpha"', terms, {2}, 10) == expected
        assert index.retrieve_expansion('"alpha"', terms, {2}, 10) == expected
        assert len(index._query_readers) == 3
        assert sum(reader._query_cache_hits for reader in index._query_readers) == 3
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index._query_readers[0].db.execute("DELETE FROM chunks")
        with sqlite3.connect(path) as writer:
            writer.execute("DELETE FROM chunk_fts WHERE rowid=1")
            writer.execute("DELETE FROM chunks WHERE id=1")
        refreshed = index.retrieve_expansion('"alpha"', terms, {2}, 10)
        monkeypatch.setattr(sag2, "QUERY_WORKERS", 1)
        assert refreshed == index.retrieve_expansion('"alpha"', terms, {2}, 10)
        assert all(row["id"] != 1 for row in refreshed[0])
        monkeypatch.setattr(sag2, "QUERY_WORKERS", 8)
        with pytest.raises(sqlite3.OperationalError):
            index.retrieve_expansion('"unterminated', terms, set(), 10)
    finally:
        readers = list(index._query_readers)
        index.close()
    for reader in readers:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            reader.db.execute("SELECT 1")


def test_cache_rebuilds_when_corpus_changes(tmp_path):
    path = tmp_path / "cache.sqlite"
    index = sag2.CorpusIndex(path, ORACLE["corpus"], "old")
    index.close()
    updated = sag2.CorpusIndex(path, [{"title": "New", "text": "New corpus"}], "new")
    try:
        assert updated.document_count == 1
        assert updated.retrieve('"new"', set(), 10)[0]["title"] == "New"
        assert updated.retrieve('"anchor"', set(), 10) == []
    finally:
        updated.close()


def test_parallel_index_rows_and_cache_match_serial(tmp_path, monkeypatch):
    corpus = [*ORACLE["corpus"], ORACLE["corpus"][0]]
    serial = sag2.CorpusIndex(tmp_path / "serial.sqlite", corpus, "fixture", index_workers=1)
    parallel_path = tmp_path / "parallel.sqlite"
    parallel = sag2.CorpusIndex(parallel_path, corpus, "fixture", index_workers=4)
    try:
        assert serial.frequencies == parallel.frequencies
        for table in ("chunks", "chunk_fts", "metadata"):
            query = f"SELECT rowid,* FROM {table} ORDER BY rowid"
            assert [tuple(r) for r in serial.db.execute(query)] == [
                tuple(r) for r in parallel.db.execute(query)
            ]
    finally:
        serial.close()
        parallel.close()
    monkeypatch.setattr(sag2, "ProcessPoolExecutor", lambda *a, **kw: pytest.fail("cache rebuilt"))
    cached = sag2.CorpusIndex(parallel_path, corpus, "fixture", index_workers=12)
    cached.close()


def test_index_workers_cli_is_independent_of_query_concurrency():
    args = benchmark.build_parser().parse_args(["--index-workers", "14", "--concurrency", "3"])
    assert args.index_workers == 14
    assert args.concurrency == 3
    assert args.chunk_chars == 2200


@pytest.mark.parametrize("text", ["甲乙😀" * 1300, "word " * 1500, "x = 1;\n" * 1000], ids=["unicode", "words", "unpunctuated-lines"])
def test_long_unpunctuated_text_stays_intact(text):
    assert sag2.chunk_text(text) == [text.rstrip("\n")]


def test_chunks_keep_complete_sentences_and_overlap():
    first, second = "a" * 1050, "b" * 1100
    assert sag2.chunk_text(first + "\n\n" + second, 1000) == [first, first + "\n\n" + second]
    assert sag2.chunk_text(first + ". " + second, 1000) == [first + ". ", first + ". " + second]


def test_default_index_splits_sentences_and_rebuilds_previous_chunk_policy(tmp_path):
    corpus = [{"title": "Long document", "text": "longword remains intact. " * 100}]
    path = tmp_path / "long.sqlite"
    index = sag2.CorpusIndex(path, corpus, "fixture")
    rows = index.db.execute("SELECT content,spans FROM chunks ORDER BY id").fetchall()
    assert len(rows) > 1
    covered = set()
    for row in rows:
        assert row[0].rstrip().endswith(".")
        assert len(row[0]) <= 2200
        for start, end in json.loads(row[1]):
            assert corpus[0]["text"][start:end] in row[0]
            covered.update(range(start, end))
    assert len(covered) == len(corpus[0]["text"])
    with index.db:
        index.db.execute("UPDATE metadata SET signature=?", (sag2.compact_json(["fixture", 1000, 2]),))
        index.db.execute("DELETE FROM chunks")
    index.close()
    rebuilt = sag2.CorpusIndex(path, corpus, "fixture")
    try:
        assert rebuilt.total == len(rows)
    finally:
        rebuilt.close()


@pytest.mark.parametrize("index_workers", [1, 4])
def test_file_index_preserves_retrieval_without_storing_text(tmp_path, index_workers):
    corpus = [
        *[dict(doc) for doc in ORACLE["corpus"]],
        {"title": "Long", "text": "anchor longtext " * 500},
    ]
    path = tmp_path / "storage.sqlite"
    full = sag2.CorpusIndex(path, corpus, "fixture")
    hits = full.retrieve('"anchor"', set(), 20)
    frequencies = dict(full.frequencies)
    full.close()
    for number, doc in enumerate(corpus):
        raw = doc["text"].replace("\n", "\r\n").encode()
        doc["path"] = f"{number}.txt"
        doc["sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / doc["path"]).write_bytes(raw)
    lean = sag2.CorpusIndex(
        path, corpus, "fixture", source_dir=tmp_path, index_workers=index_workers
    )
    try:
        assert not {"content", "full_doc"} & {
            row[1] for row in lean.db.execute("PRAGMA table_info(chunks)")
        }
        expected = [{key: value for key, value in hit.items() if key != "full_doc"} for hit in hits]
        actual = lean.retrieve('"anchor"', set(), 20)
        assert [{key: hit[key] for key in expected[0]} for hit in actual] == expected
        assert lean.frequencies == frequencies
        assert all(
            tuple(row) == (None, None, None)
            for row in lean.db.execute("SELECT title,heading,body FROM chunk_fts")
        )
        source = lean.db.execute(
            "SELECT path FROM source_files WHERE document_id=?", (actual[0]["document_id"],)
        ).fetchone()[0]
        (tmp_path / source).write_text("Source changed")
        with pytest.raises(ValueError, match="Source changed since indexing"):
            lean.retrieve('"anchor"', set(), 20)
    finally:
        lean.close()


def test_raw_passage_ranking_does_not_duplicate_title_weight(tmp_path):
    corpus = [
        {
            "title": "Anchor Anchor",
            "text": "unrelated filler word other random plain dummy extra more context",
        },
        {"title": "Body", "text": "anchor anchor anchor anchor"},
        *[{"title": f"Noise {i}", "text": "filler random dummy"} for i in range(10)],
    ]
    index = sag2.CorpusIndex(tmp_path / "raw.sqlite", corpus, "raw-passages")
    try:
        # The original raw MuSiQue import has an empty heading. Duplicating the
        # title there reverses these two candidates before the candidate cutoff.
        assert [row["title"] for row in index.retrieve('"anchor"', set(), 2)] == [
            "Body",
            "Anchor Anchor",
        ]
    finally:
        index.close()


@pytest.mark.parametrize("punctuation", ["。", "！", "？"])
def test_chinese_sentences_do_not_share_expansion_context(tmp_path, punctuation):
    content = punctuation.join(
        ["台风登陆福建，救援人员前往沿海", "成都大运会开幕", "母婴设施建设启动"]
    )
    index = sag2.CorpusIndex(
        tmp_path / "chinese-context.sqlite", [{"title": "新闻", "text": content}], "chinese"
    )
    try:
        terms = {
            term["term"]: term
            for term in index.expansion_terms(
                "台风 福建", [{"document_id": 1, "content": content, "jev_score": 1}], 100
            )
        }
        assert terms["救援"]["query_context_score"] > 0
        assert terms["大运会"]["query_context_score"] == 0
        assert terms["母婴"]["query_context_score"] == 0
    finally:
        index.close()


def test_sentence_boundaries_preserve_english_initials_and_split_chinese():
    assert sag2.SENTENCES.split("台风登陆。救援开始！人员安全吗？仍在核查") == [
        "台风登陆", "救援开始", "人员安全吗", "仍在核查"
    ]
    assert sag2.SENTENCES.split("J. R. Smith arrived. Help! Why?\nNext") == [
        "J. R. Smith arrived", "Help", "Why", "Next"
    ]


def test_sentence_context_preserves_sag2_addition_order(tmp_path):
    content = "alpha beta gamma redword. gamma beta alpha blueword."
    index = sag2.CorpusIndex(
        tmp_path / "context.sqlite", [{"title": "Context", "text": content}], "context"
    )
    try:
        # Frozen corpus statistics from a 11656-passage corpus. The two
        # sentences contain the same query words in different orders.
        index.total = 11656
        index.frequencies.update({"alpha": 1, "beta": 2, "gamma": 3})
        terms = index.expansion_terms(
            "alpha beta gamma", [{"document_id": 1, "content": content, "jev_score": 1}], 2
        )
        assert [term["term"] for term in terms] == ["redword", "blueword"]
    finally:
        index.close()


def test_batching_limits_context_without_truncating_source():
    client = OracleJev()
    candidate = {"id": 1, "title": "Large", "content": "token " * 33000}
    with pytest.raises(ValueError, match="not truncated"):
        asyncio.run(client.score("Anchor", [candidate], []))
    assert not client.payloads


def test_sag2_splits_large_round_without_request_count_limit():
    client = OracleJev()
    candidates = [{"id": i, "title": "Large", "content": "token " * 2500} for i in range(100)]
    evidence = [{"title": "Context", "content": "Shared evidence", "previous_jev_score": 0.8}]
    scored, usage = asyncio.run(client.score("Anchor", candidates, evidence))
    assert len(client.payloads) > 3
    assert usage["requests"] == len(client.payloads)
    assert all(batch["estimated_input_tokens"] <= 64000 for batch in usage["batches"])
    assert len({payload["state"] for payload in client.payloads}) == 1
    state = json.loads(client.payloads[0]["state"])
    assert state["previously_retrieved_evidence"][0]["previous_jev_score"] == 0.8
    questions = {
        key: question
        for payload in client.payloads
        for key, question in payload["questions"].items()
    }
    assert len(questions) == len(candidates)
    assert [row["id"] for row in scored] == [row["id"] for row in candidates]
    for candidate in candidates:
        assert (
            questions[f"candidate_{candidate['id']}"]["instructions"]["candidate_passage"]
            == candidate["content"]
        )


def test_jev_real_http_retries_429_and_rejects_invalid_scores(monkeypatch):
    async def scenario():
        attempts = []

        async def handler(request):
            payload = await request.json()
            attempts.append(payload)
            assert request.headers["Authorization"] == "Bearer local-test"
            if len(attempts) == 1:
                return web.Response(status=429)
            return web.json_response(
                {
                    "model": "local-jev",
                    "answers": {key: {"type": "noul", "noul": 1.1} for key in payload["questions"]},
                }
            )

        async def no_delay(_seconds):
            return None

        monkeypatch.setattr(sag2.asyncio, "sleep", no_delay)
        app = web.Application()
        app.router.add_post("/decisions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                client = sag2.JevClient(
                    session, f"http://127.0.0.1:{port}/decisions", "local-jev", "local-test"
                )
                with pytest.raises(ValueError, match="invalid noul"):
                    await client.score(
                        "Anchor", [{"id": 1, "title": "Anchor", "content": "Evidence"}], []
                    )
            assert len(attempts) == 2
            assert attempts[0] == attempts[1]
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_all_dataset_cli_checkpoint_resume_and_result_contract(tmp_path):
    corpus = [{"title": "Anchor", "text": "Anchor evidence", "idx": "source_0"}]
    rows = {
        "hotpotqa": [
            {
                "_id": "hotpot",
                "question": "Anchor?",
                "answer": "yes",
                "context": [["Anchor", ["Anchor evidence"]]],
                "supporting_facts": [["Anchor", 0]],
            }
        ],
        "2wikimultihopqa": [
            {
                "_id": "wiki",
                "question": "Anchor?",
                "answer": "yes",
                "context": [["Anchor", ["Anchor evidence"]]],
                "supporting_facts": [["Anchor", 0]],
            }
        ],
        "musique": [
            {
                "id": "musique",
                "question": "Anchor?",
                "answer": "yes",
                "paragraphs": [
                    {"title": "Anchor", "text": "Anchor evidence", "is_supporting": True}
                ],
            }
        ],
    }
    for name, samples in rows.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(samples))
        (tmp_path / f"{name}_corpus.json").write_text(json.dumps(corpus))
    args = benchmark.build_parser().parse_args(
        ["--dataset-dir", str(tmp_path), "--max-hops", "0", "--cache-dir", str(tmp_path / "cache")]
    )
    assert args.jev_question_type == "noul"
    args.jev_base_url, args.jev_model = "http://localhost/decisions", "oracle"
    client = OracleJev()
    for name in benchmark.DATASETS:
        summary = asyncio.run(benchmark.run_dataset(name, args, client, tmp_path / "output"))
        config = json.loads((tmp_path / "output" / name / "config.json").read_text())
        assert config["algorithm"] == "SAG2"
        assert config["jev_question_type"] == "noul"
        prefix = "Recall"
        assert summary["metrics"][f"{prefix}@5"] == 1
        result_path = tmp_path / "output" / name / "search_results.json"
        search_rows = json.loads(result_path.read_text())
        assert search_rows[0]["retrieved_docs"] == ["Anchor\nAnchor evidence"]
        assert search_rows[0]["dataset_sample_id"] == (rows[name][0].get("_id", "musique"))
    assert len(client.payloads) == 3
    for name in benchmark.DATASETS:
        asyncio.run(benchmark.run_dataset(name, args, client, tmp_path / "output"))
    assert len(client.payloads) == 3
    args.expansion_term_limit = 1
    with pytest.raises(ValueError, match="resume settings/data/code differ"):
        asyncio.run(benchmark.run_dataset("musique", args, client, tmp_path / "output"))


def test_sag2_499_retries_count_only_the_successful_request(monkeypatch):
    async def scenario():
        attempts = []

        async def handler(request):
            payload = await request.json()
            attempts.append(payload)
            if len(attempts) <= 5:
                return web.Response(status=499)
            return web.json_response(
                {
                    "model": "local-jev",
                    "answers": {key: {"type": "noul", "noul": 0.7} for key in payload["questions"]},
                }
            )

        async def no_delay(_seconds):
            pass

        monkeypatch.setattr(sag2.asyncio, "sleep", no_delay)
        app = web.Application()
        app.router.add_post("/decisions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                client = sag2.JevClient(
                    session, f"http://127.0.0.1:{port}/decisions", "local-jev", "local-test"
                )
                _, usage = await client.score(
                    "Question", [{"id": 1, "title": "Evidence", "content": "Source text"}], []
                )
            assert len(attempts) == 6 and all(payload == attempts[0] for payload in attempts)
            assert usage["requests"] == 1
            assert usage["batches"][0]["http_attempts"] == 6
            assert usage["batches"][0]["failed_attempts"] == {"http_499": 5}
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_reopening_index_reuses_persisted_statistics(tmp_path, monkeypatch):
    path = tmp_path / "reuse.sqlite"
    index = sag2.CorpusIndex(path, ORACLE["corpus"], "fixture")
    expected = {
        name: getattr(index, name)
        for name in ("frequencies", "total", "document_count", "title_phrases", "aliases", "diversity_idf")
    }
    expected_hits = index.retrieve('"bridge"', set(), 10)
    index.close()

    def unexpected(*args, **kwargs):
        pytest.fail("Reopening an unchanged index reprocessed corpus text")

    with monkeypatch.context() as patch:
        patch.setattr(sag2, "chunk_document", unexpected)
        patch.setattr(sag2, "diversity_tokens", unexpected)
        patch.setattr(sag2, "tokenize", unexpected)
        cached = sag2.CorpusIndex(path, ORACLE["corpus"], "fixture")
    try:
        for name, value in expected.items():
            assert getattr(cached, name) == value
        assert cached.retrieve('"bridge"', set(), 10) == expected_hits
    finally:
        cached.close()

    changed = sag2.CorpusIndex(path, [{"title": "New", "text": "New corpus"}], "new")
    try:
        assert changed.document_count == 1
        assert "anchor" not in changed.diversity_idf
        assert changed.retrieve('"bridge"', set(), 10) == []
    finally:
        changed.close()
