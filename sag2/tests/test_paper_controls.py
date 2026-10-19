import asyncio

import pytest


protocol = __import__("sag2_controls")
sag2 = __import__("sag2")


class FinalScorer:
    def __init__(self):
        self.calls = []

    async def score(self, query, candidates, evidence):
        self.calls.append((query, candidates, evidence))
        return [
            {**row, "jev_score": 0.9 if row["title"] == "Second" else 0.1} for row in candidates
        ], {"judgments": len(candidates)}


@pytest.fixture
def feedback_index(tmp_path):
    corpus = [
        {"title": "First", "text": "queryword bridge bridge otherword"},
        {"title": "Second", "text": "bridge bridge bridge"},
        *[{"title": f"Noise {i}", "text": f"noise common filler item {i}"} for i in range(30)],
    ]
    index = sag2.CorpusIndex(tmp_path / "fixture.sqlite", corpus, "fixture")
    yield index
    index.close()


def test_rm3_weights_match_worked_feedback_example(feedback_index):
    feedback = protocol.FeedbackTerms(feedback_index)
    weights, trace = feedback.expand("queryword")
    # One feedback paragraph: bridge TF2, queryword TF1, otherword TF1.
    # RM3 original-query weight is 0.5, giving these literal interpolated values.
    assert weights == pytest.approx({"queryword": 0.625, "bridge": 0.25, "otherword": 0.125})
    assert trace["feedback_documents"][0]["normalizer"] == 4
    assert trace["feedback_documents"][0]["positive_bm25"] > 0


def test_weighted_bm25_applies_term_weights_before_truncation(feedback_index):
    hits = protocol.weighted_bm25(feedback_index, {"bridge": 0.001, "queryword": 0.999}, 1)
    assert hits[0]["title"] == "First"
    hits = protocol.weighted_bm25(feedback_index, {"bridge": 1.0}, 1)
    assert hits[0]["title"] == "Second"


def test_rm3_pair_has_no_intermediate_model_calls_and_same_candidate_pool(feedback_index):
    feedback = protocol.FeedbackTerms(feedback_index)
    client = FinalScorer()
    pure = asyncio.run(
        protocol.run_protocol(
            feedback_index, client, "queryword", "rm3", candidate_k=20, feedback=feedback
        )
    )
    assert client.calls == []
    final = asyncio.run(
        protocol.run_protocol(
            feedback_index, client, "queryword", "rm3_jev", candidate_k=20, feedback=feedback
        )
    )
    assert len(client.calls) == 1
    assert client.calls[0][2] == []
    assert {row["id"] for row in pure["candidates"]} == {row["id"] for row in final["candidates"]}
    assert final["candidates"][0]["title"] == "Second"


def test_lexical_expansion_uses_uniform_weights_and_only_one_final_score(feedback_index):
    client = FinalScorer()
    result = asyncio.run(
        protocol.lexical_expansion(
            feedback_index, client, "queryword", candidate_k=20, seed_k=1, term_limit=5, final_k=30
        )
    )
    assert len(client.calls) == 1
    assert client.calls[0][2] == []
    assert {row["title"] for row in client.calls[0][1]} >= {"First", "Second"}
    assert all(
        term["seed_weight"] == 1 for stage in result["trace"] for term in stage["term_scores"]
    )


def test_gar_expands_with_bm25_and_model_is_only_used_at_the_end(feedback_index):
    graph = {identifier: [] for identifier in range(1, feedback_index.total + 1)}
    graph[1] = [{"id": 2, "bm25_score": -1.0}]
    client = FinalScorer()
    pure = asyncio.run(
        protocol.gar_search(feedback_index, graph, None, "queryword", candidate_k=2, batch_size=1)
    )
    assert [stage["source"] for stage in pure["trace"]] == ["initial", "graph"]
    assert pure["trace"][1]["parent_ids"] == {2: 1}
    assert pure["candidates"][1]["score"] == 0  # No original query term in the bridge passage.
    final = asyncio.run(
        protocol.gar_search(feedback_index, graph, client, "queryword", candidate_k=2, batch_size=1)
    )
    assert len(client.calls) == 1
    assert client.calls[0][2] == []
    assert {row["id"] for row in pure["candidates"]} == {row["id"] for row in final["candidates"]}
    assert final["candidates"][0]["title"] == "Second"


def test_final_candidate_limit_preserves_new_expansion_evidence(tmp_path):
    corpus = [
        {"title": "First", "text": "queryword " * 10 + "connectword"},
        *[{"title": f"Initial {i}", "text": "queryword"} for i in range(9)],
        *[{"title": f"Bridge {i}", "text": "connectword secondword"} for i in range(10)],
        *[{"title": f"Proof {i}", "text": "secondword"} for i in range(10)],
    ]
    index = sag2.CorpusIndex(tmp_path / "final-limit.sqlite", corpus, "final-limit")
    try:
        scorer = FinalScorer()
        result = asyncio.run(
            protocol.lexical_expansion(
                index, scorer, "queryword", candidate_k=10, seed_k=1, final_k=10
            )
        )
        assert len(scorer.calls) == 1
        assert len(scorer.calls[0][1]) == 10
        assert any(row["title"].startswith("Proof") for row in scorer.calls[0][1])
        assert len(set(result["final_scoring_candidate_ids"])) == 10
    finally:
        index.close()


def test_gar_pure_top10_equals_initial_bm25_when_initial_batch_contains_top10(tmp_path):
    corpus = [
        {"title": f"Item {i}", "text": "queryword " * (30 - i) + f"other{i}"} for i in range(25)
    ]
    index = sag2.CorpusIndex(tmp_path / "gar.sqlite", corpus, "gar-fixture")
    graph = {
        identifier: [{"id": identifier % 25 + 1, "bm25_score": -0.1}] for identifier in range(1, 26)
    }
    try:
        initial = index.retrieve(sag2.match_query("queryword"), set(), 10)
        result = asyncio.run(protocol.gar_search(index, graph, None, "queryword", candidate_k=20))
        assert [row["id"] for row in result["candidates"][:10]] == [row["id"] for row in initial]
    finally:
        index.close()


def test_graph_is_full_text_cached_and_excludes_self(feedback_index, tmp_path):
    path = tmp_path / "graph.sqlite"
    graph, first = protocol.build_bm25_graph(feedback_index, path, "fixture", neighbors=2)
    assert first["new_source_queries"] == feedback_index.total
    assert 2 in [edge["id"] for edge in graph[1]]
    assert all(source not in [edge["id"] for edge in edges] for source, edges in graph.items())
    second_graph, second = protocol.build_bm25_graph(feedback_index, path, "fixture", neighbors=2)
    assert graph == second_graph
    assert second["new_source_queries"] == 0
    with pytest.raises(ValueError, match="different corpus"):
        protocol.build_bm25_graph(feedback_index, path, "changed", neighbors=2)
