import asyncio
import json

import aiohttp
import pytest

from local_reranker import LocalScorer, STANDARD_INSTRUCTION
from scripts import run_sag2_benchmark as benchmark


class FixtureModel:
    def __init__(self, *args):
        self.model = "jina"
        self.config = {"model": "jina", "weights_sha256": "fixed-local-model"}
        self.device = "cpu"
        self.lock = asyncio.Lock()
        self.requests = []

    def rank(self, query, documents, instruction):
        self.requests.append((query, documents, instruction))
        return {
            "model": "local/jina/test",
            "results": [
                {"index": i, "relevance_score": (i + 1) / len(documents)}
                for i in range(len(documents))
            ],
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }


def test_local_scorer_keeps_context_maps_scores_and_replays(tmp_path):
    async def scenario():
        engine = FixtureModel()
        scorer = LocalScorer(engine, tmp_path)
        candidates = [{"id": 5, "title": "Full title", "content": "Full source text"}]
        evidence = [
            {"title": "Earlier", "content": "Earlier source text", "previous_jev_score": 0.8}
        ]
        first, usage = await scorer.score("Original question", candidates, evidence)
        again, reused = await scorer.score("Original question", candidates, evidence)
        assert len(engine.requests) == 1
        state = json.loads(engine.requests[0][0])
        assert state["original_question"] == "Original question"
        assert state["previously_retrieved_evidence"][0]["previous_jev_score"] == 0.8
        assert "evidence_score_note" in state
        assert engine.requests[0][1] == ["Full title\nFull source text"]
        assert first == again
        assert first[0]["jev_score"] == 1
        assert not usage["batches"][0]["reused"]
        assert reused["batches"][0]["reused"]

    asyncio.run(scenario())


@pytest.mark.parametrize("protocol,hops", [("sag2", 2), ("standard", 0)])
def test_local_benchmark_no_network_or_credentials_and_resume(
    tmp_path, monkeypatch, protocol, hops
):
    import local_reranker

    engine = FixtureModel()
    monkeypatch.setattr(local_reranker, "LocalReranker", lambda *args: engine)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Local mode must not make network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    for key in ("OPENROUTER_API_KEY", "JEV_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    corpus = [
        {"title": f"Passage {i}", "text": f"Anchor bridge necessary fact {i}."} for i in range(24)
    ]
    (tmp_path / "hotpotqa_corpus.json").write_text(json.dumps(corpus))
    (tmp_path / "hotpotqa.json").write_text(
        json.dumps(
            [
                {
                    "_id": "local-question",
                    "question": "Anchor?",
                    "supporting_facts": [["Passage 0", 0]],
                    "context": [["Passage 0", ["Anchor bridge necessary fact 0."]]],
                }
            ]
        )
    )
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "local",
            "--model-path",
            str(tmp_path),
            "--dataset",
            "hotpotqa",
            "--dataset-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "results"),
            "--index-workers",
            "1",
            "--env-file",
            str(tmp_path / "absent.env"),
            "--rerank-protocol",
            protocol,
            "--expansions",
            str(hops),
        ]
    )
    asyncio.run(benchmark.run(args))
    summary = json.loads((tmp_path / "results/hotpotqa/metrics.json").read_text())
    assert summary["algorithm"] == "Local SAG2"
    assert (
        summary["jev_requests"]
        == summary["model_api_requests"]
        == summary["model_api_cost_usd"]
        == 0
    )
    assert summary["actual_reranker_models"] == ["local/jina/test"]
    record = json.loads((tmp_path / "results/hotpotqa/details.jsonl").read_text())
    assert "reranker_usage" in record and "jev_usage" not in record
    if protocol == "standard":
        assert len(engine.requests) == 1
        assert engine.requests[0][0] == "Anchor?"
        assert engine.requests[0][2] == STANDARD_INSTRUCTION
        assert summary["requests"] == 1
    else:
        assert 1 <= len(engine.requests) <= 3
    count = len(engine.requests)
    args.resume, args.output_dir = tmp_path / "results", None
    asyncio.run(benchmark.run(args))
    assert len(engine.requests) == count


def test_invalid_local_score_fails(tmp_path):
    async def scenario():
        engine = FixtureModel()
        engine.rank = lambda *args: {
            "model": "broken",
            "results": [{"index": 0, "relevance_score": float("nan")}],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        with pytest.raises(ValueError, match="Invalid local reranker score"):
            await LocalScorer(engine, tmp_path).score("query", [{"title": "A", "content": "B"}], [])

    asyncio.run(scenario())
