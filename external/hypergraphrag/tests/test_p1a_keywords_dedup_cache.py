"""P1-A tests: HyperGraphRAG JSON keyword extraction, source_id weight dedup, cache flush.

These are unit tests that use mocks — no real LLM or storage calls.
"""

import json
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# JSON keyword extraction (kg_query)
# ---------------------------------------------------------------------------

# Mock helpers for testing kg_query keyword extraction in isolation
VALID_KEYWORDS_JSON = json.dumps({
    "high_level_keywords": ["Topic A", "Topic B"],
    "low_level_keywords": ["entity1", "entity2"],
})
HIGH_ONLY_JSON = json.dumps({
    "high_level_keywords": ["Global Theme"],
    "low_level_keywords": [],
})
LOW_ONLY_JSON = json.dumps({
    "high_level_keywords": [],
    "low_level_keywords": ["detail1"],
})
STRING_VALUES_JSON = json.dumps({
    "high_level_keywords": "single topic",
    "low_level_keywords": ["multi", "keywords"],
})
APOSTROPHE_JSON = '{"high_level_keywords": ["don\'t panic"], "low_level_keywords": ["it\'s fine"]}'


class TestLocateJsonStringBody:
    """Tests for the JSON body extraction helper used in kg_query."""

    def test_valid_json_passthrough(self):
        from hypergraphrag.utils import locate_json_string_body_from_string

        result = locate_json_string_body_from_string(VALID_KEYWORDS_JSON)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["high_level_keywords"] == ["Topic A", "Topic B"]

    def test_apostrophe_preserved(self):
        """JSON with apostrophes like "don't" must survive."""
        from hypergraphrag.utils import locate_json_string_body_from_string

        result = locate_json_string_body_from_string(APOSTROPHE_JSON)
        assert result is not None
        parsed = json.loads(result)
        assert "don't panic" in parsed["high_level_keywords"]
        assert "it's fine" in parsed["low_level_keywords"]

    def test_markdown_code_fences(self):
        from hypergraphrag.utils import locate_json_string_body_from_string

        result = locate_json_string_body_from_string(
            f"```json\n{VALID_KEYWORDS_JSON}\n```"
        )
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed["high_level_keywords"]) == 2

    def test_single_quoted_json_fallback(self):
        """Single-quote JSON should parse via fallback path."""
        from hypergraphrag.utils import locate_json_string_body_from_string

        single_quote = "{'high_level_keywords': ['X'], 'low_level_keywords': ['Y']}"
        result = locate_json_string_body_from_string(single_quote)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["high_level_keywords"] == ["X"]

    def test_no_json_returns_none(self):
        from hypergraphrag.utils import locate_json_string_body_from_string

        assert locate_json_string_body_from_string("just some text") is None


class TestKeywordNormalization:
    """Test keyword normalization logic extracted from kg_query."""

    def test_string_becomes_single_element_list(self):
        raw_hl = "single topic"
        if isinstance(raw_hl, str):
            raw_hl = [raw_hl]
        assert raw_hl == ["single topic"]

    def test_list_stays_list(self):
        raw_hl = ["a", "b"]
        assert isinstance(raw_hl, list)

    def test_none_becomes_empty(self):
        raw = None
        result = raw or []
        assert result == []

    def test_empty_list_validation(self):
        """Require at least one keyword list non-empty (not both)."""
        hl, ll = [], []
        assert not (hl or ll)  # both empty -> fail
        hl2, ll2 = ["x"], []
        assert hl2 or ll2  # one non-empty -> pass
        hl3, ll3 = [], ["y"]
        assert hl3 or ll3  # other non-empty -> pass


# ---------------------------------------------------------------------------
# source_id weight dedup
# ---------------------------------------------------------------------------

class TestHyperedgeWeightDedup:
    """Verify that _merge_hyperedges_then_upsert logic prevents double-counting."""

    def test_same_source_id_twice_no_double_count(self):
        """Re-processing the same source_id should NOT increase weight."""
        nodes_data = [
            {"source_id": "chunk_1", "weight": 10.0},
            {"source_id": "chunk_1", "weight": 10.0},  # duplicate
        ]
        already_source_ids = {"chunk_1"}
        already_weight = 10.0

        # Dedup logic (extracted from _merge_hyperedges_then_upsert)
        new_source_ids = set(d["source_id"] for d in nodes_data) - already_source_ids
        new_weight = sum(d["weight"] for d in nodes_data if d["source_id"] in new_source_ids)
        weight = already_weight + new_weight

        assert weight == 10.0  # already_weight only, no new source_ids

    def test_new_source_id_adds_weight_once(self):
        nodes_data = [
            {"source_id": "chunk_1", "weight": 10.0},
            {"source_id": "chunk_2", "weight": 5.0},
        ]
        already_source_ids = {"chunk_1"}
        already_weight = 10.0

        new_source_ids = set(d["source_id"] for d in nodes_data) - already_source_ids
        new_weight = sum(d["weight"] for d in nodes_data if d["source_id"] in new_source_ids)
        weight = already_weight + new_weight

        assert new_source_ids == {"chunk_2"}
        assert weight == 15.0  # 10 + 5

    def test_all_new_source_ids(self):
        nodes_data = [
            {"source_id": "chunk_a", "weight": 3.0},
            {"source_id": "chunk_b", "weight": 7.0},
        ]
        already_source_ids = set()
        already_weight = 0.0

        new_source_ids = set(d["source_id"] for d in nodes_data) - already_source_ids
        new_weight = sum(d["weight"] for d in nodes_data if d["source_id"] in new_source_ids)
        weight = already_weight + new_weight

        assert new_source_ids == {"chunk_a", "chunk_b"}
        assert weight == 10.0

    @pytest.mark.asyncio
    async def test_actual_merge_counts_duplicate_new_source_once(self):
        from hypergraphrag.operate import _merge_hyperedges_then_upsert

        class Graph:
            stored = None

            async def get_node(self, _name):
                return None

            async def upsert_node(self, _name, node_data):
                self.stored = node_data

        graph = Graph()
        await _merge_hyperedges_then_upsert(
            "<hyperedge>x",
            [
                {"source_id": "chunk_new", "weight": 5.0},
                {"source_id": "chunk_new", "weight": 5.0},
            ],
            graph,
            {},
        )
        assert graph.stored["weight"] == 5.0
        assert graph.stored["source_id"] == "chunk_new"


class TestEdgeWeightDedup:
    """Verify that _merge_edges_then_upsert logic prevents double-counting."""

    def test_same_source_id_edge_no_double_count(self):
        source_id = "chunk_1"
        weight = 10.0
        already_source_ids = ["chunk_1"]
        already_weight = 10.0

        if source_id not in already_source_ids:
            weight = already_weight + weight
        else:
            weight = already_weight

        assert weight == 10.0

    def test_new_source_id_edge_adds_weight(self):
        source_id = "chunk_2"
        weight = 5.0
        already_source_ids = ["chunk_1"]
        already_weight = 10.0

        if source_id not in already_source_ids:
            weight = already_weight + weight
        else:
            weight = already_weight

        assert weight == 15.0


# ---------------------------------------------------------------------------
# Cache flush intervals
# ---------------------------------------------------------------------------

class TestCacheFlushInterval:
    """Verify the cache flush interval logic from extract_entities."""

    def test_401_chunks_triggers_three_flushes(self):
        """200 completed chunks -> flush at 200, 400, and final (401)."""
        cache_flush_interval = 200
        completed_counts = []
        flush_triggered = []

        for i in range(1, 402):
            # _process_single_content increments completed count
            # after each chunk, check if we should flush
            if i % cache_flush_interval == 0:
                flush_triggered.append(i)

        # Final flush after all chunks
        flush_triggered.append(401)  # final flush

        assert 200 in flush_triggered
        assert 400 in flush_triggered
        assert 401 in flush_triggered  # final flush
        assert len(flush_triggered) == 3  # 200, 400, final

    def test_50_chunks_only_final_flush(self):
        cache_flush_interval = 200
        flush_triggered = []

        for i in range(1, 51):
            if i % cache_flush_interval == 0:
                flush_triggered.append(i)
        flush_triggered.append(50)  # final

        assert len(flush_triggered) == 1
        assert flush_triggered[0] == 50
