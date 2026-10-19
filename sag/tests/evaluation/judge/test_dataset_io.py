"""Tests for explicit dataset adapters and the canonical ground-truth repository."""

import json

import pytest

from pipeline.evaluation.judge.adapters.base import ensure_unique_questions
from pipeline.evaluation.judge.dataset_adapters.errors import (
    DatasetCapabilityError,
    DatasetSchemaError,
    UnsupportedDatasetError,
)
from pipeline.evaluation.judge.dataset_adapters.models import DatasetCapability
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository, load_evidence_map
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    AmbiguousGroundTruthMatchError,
)


def _hotpot_row():
    return {
        "question": "Q",
        "answer": "A",
        "supporting_facts": [["Alpha", 1], ["Beta", 0]],
        "context": [
            ["Alpha", ["not evidence", "alpha evidence"]],
            ["Beta", ["beta evidence"]],
        ],
    }


def test_load_evidence_map_resolves_hotpot_supporting_facts(tmp_path):
    (tmp_path / "hotpotqa.json").write_text(json.dumps([_hotpot_row()]), encoding="utf-8")
    assert load_evidence_map("hotpotqa", str(tmp_path)) == {
        0: ["Alpha\nnot evidencealpha evidence", "Beta\nbeta evidence"]
    }


def test_load_evidence_map_uses_declared_sample_schema(tmp_path):
    dataset = [
        {
            "question": "Q",
            "answer": "A",
            "paragraphs": [
                {"title": "T1", "text": "explicit", "is_supporting": True},
                {"title": "T2", "text": "noise", "is_supporting": False},
            ],
        }
    ]
    (tmp_path / "sample.json").write_text(json.dumps(dataset), encoding="utf-8")
    assert load_evidence_map("sample", str(tmp_path)) == {0: ["T1\nexplicit"]}


def test_load_evidence_map_ignores_2wiki_relation_triples(tmp_path):
    row = {
        "question": "Q",
        "answer": "A",
        "evidences": [["Lothair II", "mother", "Ermengarde of Tours"]],
        "supporting_facts": [["Lothair II", 0]],
        "context": [["Lothair II", ["gold sentence"]]],
    }
    (tmp_path / "2wikimultihopqa.json").write_text(json.dumps([row]), encoding="utf-8")
    assert load_evidence_map("2wikimultihopqa", str(tmp_path)) == {0: ["Lothair II\ngold sentence"]}


def test_load_evidence_map_resolves_musique_supporting_paragraphs(tmp_path):
    dataset = [
        {
            "question": "Q",
            "answer": "A",
            "paragraphs": [
                {"title": "T1", "paragraph_text": "supporting fact A", "is_supporting": True},
                {"title": "T2", "paragraph_text": "distractor B", "is_supporting": False},
                {"title": "T3", "paragraph_text": "supporting fact C", "is_supporting": True},
            ],
        }
    ]
    (tmp_path / "musique.json").write_text(json.dumps(dataset), encoding="utf-8")
    assert load_evidence_map("musique", str(tmp_path)) == {
        0: ["supporting fact A", "supporting fact C"]
    }


def test_musique_rejects_empty_supporting_paragraph(tmp_path):
    dataset = [
        {
            "question": "Q",
            "answer": "A",
            "paragraphs": [{"title": "T", "paragraph_text": "", "is_supporting": True}],
        }
    ]
    (tmp_path / "musique.json").write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="must not be empty"):
        GroundTruthRepository(tmp_path).load("musique")


class TestEnsureUniqueQuestions:
    def test_no_duplicates_passes(self):
        rows = [{"question": "Q1"}, {"question": "Q2"}]
        ensure_unique_questions(rows, "test")

    def test_duplicate_raises(self):
        rows = [{"question": "Q1"}, {"question": "Q2"}, {"question": "Q1"}]
        with pytest.raises(AdapterConversionError, match="duplicate question at rows 0 and 2"):
            ensure_unique_questions(rows, "test")

    def test_error_includes_row_indices(self):
        rows = [{"question": "A"}, {"question": "B"}, {"question": "C"}, {"question": "B"}]
        with pytest.raises(AdapterConversionError) as exc_info:
            ensure_unique_questions(rows, "test")
        assert "rows 1 and 3" in str(exc_info.value)


class TestGroundTruthRepository:
    def test_timestamp_file_uses_explicit_adapter(self, tmp_path):
        (tmp_path / "hotpotqa_20240101_120000.json").write_text(
            json.dumps([_hotpot_row()]), encoding="utf-8"
        )
        repo = GroundTruthRepository(tmp_path)
        assert repo.evidence_map("hotpotqa_20240101_120000")[0] == (
            "Alpha\nnot evidencealpha evidence",
            "Beta\nbeta evidence",
        )

    def test_timestamp_name_resolves_base_file(self, tmp_path):
        base_file = tmp_path / "hotpotqa.json"
        base_file.write_text(json.dumps([_hotpot_row()]), encoding="utf-8")
        repo = GroundTruthRepository(tmp_path)
        assert repo.resolve_dataset_path("hotpotqa_20240101_120000") == base_file

    def test_narrativeqa_has_no_evidence_capability(self, tmp_path):
        (tmp_path / "narrativeqa.json").write_text(
            json.dumps([{"question": "Q", "answer": "A"}]), encoding="utf-8"
        )
        repo = GroundTruthRepository(tmp_path)
        assert repo.evidence_map("narrativeqa") == {0: ()}
        with pytest.raises(DatasetCapabilityError, match="evidence_recall"):
            repo.require_capability("narrativeqa", DatasetCapability.EVIDENCE_RECALL)

    def test_unknown_dataset_fails_explicitly(self, tmp_path):
        (tmp_path / "unknown.json").write_text(json.dumps([]), encoding="utf-8")
        with pytest.raises(UnsupportedDatasetError, match="unknown"):
            GroundTruthRepository(tmp_path).load("unknown")

    def test_numeric_suffix_is_not_treated_as_an_adapter_alias(self, tmp_path):
        (tmp_path / "narrativeqa1.json").write_text(
            json.dumps([{"question": "Q", "answer": "A"}]), encoding="utf-8"
        )
        with pytest.raises(UnsupportedDatasetError, match="narrativeqa1"):
            GroundTruthRepository(tmp_path).load("narrativeqa1")

    def test_duplicate_question_never_selects_the_first_row(self, tmp_path):
        (tmp_path / "sample.json").write_text(
            json.dumps(
                [
                    {"question": "Q", "answer": "A", "paragraphs": []},
                    {"question": "Q", "answer": "B", "paragraphs": []},
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(AmbiguousGroundTruthMatchError, match="0, 1"):
            GroundTruthRepository(tmp_path).match_question("sample", "Q")

    def test_dataset_sample_id_disambiguates_duplicate_musique_questions(self, tmp_path):
        rows = [
            {
                "id": "sample-a",
                "question": "duplicate question",
                "answer": "A",
                "paragraphs": [],
            },
            {
                "id": "sample-b",
                "question": "duplicate question",
                "answer": "B",
                "paragraphs": [],
            },
        ]
        (tmp_path / "musique.json").write_text(json.dumps(rows), encoding="utf-8")
        repo = GroundTruthRepository(tmp_path)

        assert repo.match_dataset_sample_id("musique", "sample-b", "duplicate question").id == 1
        with pytest.raises(AmbiguousGroundTruthMatchError, match="0, 1"):
            repo.match_question("musique", "duplicate question")
