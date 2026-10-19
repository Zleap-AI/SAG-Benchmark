import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from enterprise_index import ReadOnlyIndex

from scripts import run_enterprise as runner

sag2 = __import__("sag2")


def test_one_request_per_worker_caps_20_workers_at_20_requests(monkeypatch, tmp_path):
    activity = {"active": 0, "peak": 0}

    async def request(self, payload):
        activity["active"] += 1
        activity["peak"] = max(activity["peak"], activity["active"])
        await asyncio.sleep(0.01)
        activity["active"] -= 1
        return {"usage": {"cost": 0.01}}

    monkeypatch.setattr(runner.JevClient, "request", request)

    async def execute():
        clients = [runner.RecordedJev("test-key", tmp_path / str(i), {}) for i in range(20)]
        await asyncio.gather(*(client.request({}) for client in clients for _ in range(2)))
        assert all(client.costs == [0.01, 0.01] for client in clients)
        assert len(list(tmp_path.glob("*/round_*/response.json"))) == 40

    asyncio.run(execute())
    assert activity["peak"] == 20
    assert activity["active"] == 0


def test_sag2_file_index_matches_corpus_index_and_shares_metadata(tmp_path, monkeypatch):
    oracle = json.loads(Path(__file__).with_name("sag2_reference.json").read_text())
    corpus = [dict(row) for row in oracle["corpus"]]
    for number, row in enumerate(corpus):
        row["path"] = f"dsid_{number}__{row['title'].replace(' ', '-')}.txt"
        raw = row["text"].encode()
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / row["path"]).write_bytes(raw)
    path = tmp_path / "index.sqlite"
    original = sag2.CorpusIndex(path, corpus, "fixture", source_dir=tmp_path)
    signature = original.db.execute("SELECT signature FROM metadata").fetchone()[0]
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    index = ReadOnlyIndex(path, signature, tmp_path)

    def make_client():
        client = sag2.JevClient(None, "http://offline/decisions", "fixture", "unused")
        client.request = AsyncMock(
            side_effect=lambda payload: {
                "model": "fixture",
                "answers": {
                    key: {"type": "noul", "noul": (int(key.split("_")[1]) % 10 + 1) / 10}
                    for key in payload["questions"]
                },
            }
        )
        return client

    try:
        assert index.title_phrases == original.title_phrases
        assert index.aliases == original.aliases
        assert index.diversity_idf == original.diversity_idf
        clients = [make_client(), make_client()]
        results = [
            asyncio.run(
                sag2.search_sag2(current, client, "anchor multiple", candidate_k=5, seed_k=2)
            )
            for current, client in zip((original, index), clients)
        ]
        for result in results:
            result.pop("elapsed_ms")
        assert results[0] == results[1]
        assert clients[0].request.call_args_list == clients[1].request.call_args_list
        assert len(clients[0].request.call_args_list) <= 3
        assert results[1]["algorithm"] == "SAG2"
        shared = ReadOnlyIndex(
            path,
            signature,
            tmp_path,
            (index.frequencies, index.total, index.document_count),
            lexical_statistics=index.lexical_statistics,
        )
        assert shared.diversity_idf is index.diversity_idf
        shared.close()
        with monkeypatch.context() as patch:
            patch.setattr(
                Path, "read_bytes", lambda _: pytest.fail("cached startup read source text")
            )
            cached = ReadOnlyIndex(
                path, signature, tmp_path, ({}, index.total, index.document_count)
            )
            assert cached.diversity_idf == original.diversity_idf
            cached.close()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        metadata = json.loads(path.with_suffix(".lexical-statistics.json").read_text())
        assert set(metadata) == {
            "index_signature",
            "document_count",
            "title_phrases",
            "aliases",
            "diversity_idf",
        }
    finally:
        index.close()
        original.close()
