"""Offline protocol tests; the HTTP server is local and makes no paid calls."""

import asyncio
import json
from unittest.mock import patch

import aiohttp
from aiohttp import web
from scripts import run_paper_controls as run


async def verify(directory):
    calls = []
    statuses = iter([499, 500, 200])

    async def handler(request):
        payload = await request.json()
        calls.append(payload)
        status = next(statuses)
        if status != 200:
            return web.json_response({"error": "test failure"}, status=status)
        return web.json_response(
            {
                "model": run.ACTUAL_MODEL,
                "answers": {key: {"type": "noul", "noul": 0.7} for key in payload["questions"]},
                "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.01},
            }
        )

    app = web.Application()
    app.router.add_post("/decisions", handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with aiohttp.ClientSession() as session:
        client = run.Client(
            session,
            "test",
            directory / "question",
            asyncio.Semaphore(20),
            directory / "model.json",
            success_cap=1,
            no_context=True,
        )
        client.endpoint = f"http://127.0.0.1:{port}/decisions"
        candidate = {
            "id": 1,
            "title": "Anchor",
            "content": "Query anchor bridge.",
            "fusion_score": 0.1,
            "full_doc": "Anchor\nQuery anchor bridge.",
            "document_id": 1,
        }

        async def no_wait(seconds):
            pass

        with patch.object(run.asyncio, "sleep", no_wait):
            scored, usage = await client.score(
                "Query", [candidate], [{**candidate, "previous_jev_score": 0.9}]
            )
        assert scored[0]["jev_score"] == 0.7 and usage["judgments"] == 1
        assert len(calls) == 3 and len(client.successes) == 1
        assert all(json.loads(p["state"])["previously_retrieved_evidence"] == [] for p in calls)
        assert [
            json.loads(p.read_text())["status"]
            for p in sorted((directory / "question").glob("requests/*/attempt-*.json"))
        ] == [499, 500, 200]
        await client.score("Query", [candidate], [])
        assert len(calls) == 3, "Successful response was not reused"
        try:
            await client.score("Changed question", [candidate], [])
        except ValueError as error:
            assert "budget exhausted" in str(error)
        else:
            raise AssertionError("Successful request cap was not enforced")
        client.success_cap = 3
        candidates = [
            {**candidate, "id": i, "content": "ordinary evidence " * 100} for i in range(340)
        ]
        batches = client.pack("Query", candidates)
        assert sum(map(len, batches)) == 340 and len(batches) <= 3
        oversized = [{**c, "content": "ordinary evidence " * 180} for c in candidates]
        try:
            client.pack("Query", oversized)
        except ValueError as error:
            assert "success cap" in str(error)
        else:
            raise AssertionError("Token packing exceeded successful request cap")
    await server.cleanup()

    corpus = [{"title": f"Title {i}", "text": f"queryword bridge evidence {i}"} for i in range(20)]
    index = run.sag2.CorpusIndex(directory / "index.sqlite", corpus, "fixture", 2200)
    candidates = [
        {**row, "jev_score": 1 - i / 100, "fusion_score": 1 / (61 + i)}
        for i, row in enumerate(index.retrieve(index.match_query("queryword"), set(), 20))
    ]
    ranked = run.final_rank(candidates, index)
    assert ranked[0]["id"] == candidates[0]["id"]
    assert {c["id"] for c in ranked[:10]} == {c["id"] for c in candidates[:10]}
    assert ranked[10:] == candidates[10:]
    index.close()


def test_paper_client_retry_cache_budget_and_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "OUT", tmp_path / "output")
    asyncio.run(verify(tmp_path))
