"""Focused tests for the core SAG2 Judge adapter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.evaluation.judge.adapters.registry import AdapterRegistry
from pipeline.evaluation.judge.adapters.sag import SAGAdapter
from pipeline.evaluation.judge.artifacts import ArtifactLayoutResolver, JudgeArtifactStore
from pipeline.evaluation.judge.conversion import PredictionConversionService
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import SourceRunNotFoundError
from pipeline.evaluation.judge.models import ConversionRequest


def _request(root: Path, dataset: str = "musique") -> ConversionRequest:
    return ConversionRequest(
        project="sag",
        dataset=dataset,
        input_root=root,
        dataset_dir=root,
        artifact_run_root=root,
    )


def _write_sag_run(root: Path, run_id: str, rows: list[dict], qa_rows: list[dict]) -> Path:
    run_root = root / "musique" / "sag2" / run_id
    qa_root = run_root / "qa_20260101_010101"
    qa_root.mkdir(parents=True)
    (run_root / "search_results.json").write_text(json.dumps(rows), encoding="utf-8")
    (qa_root / "qa_results.json").write_text(json.dumps({"per_example": qa_rows}), encoding="utf-8")
    return run_root


def test_locate_source_defaults_to_latest_sag2_run(tmp_path):
    rows = [{"question_index": 1, "question": "Q", "retrieved_docs": ["C"]}]
    qa_rows = [{"question_index": 1, "question": "Q", "predicted_answer": "A"}]
    _write_sag_run(tmp_path, "20260101_000001", rows, qa_rows)
    latest = _write_sag_run(tmp_path, "20260102_000001", rows, qa_rows)
    incomplete = tmp_path / "musique" / "sag2" / "20260103_000001"
    incomplete.mkdir()
    (incomplete / "search_results.json").write_text(json.dumps(rows), encoding="utf-8")

    source = SAGAdapter().locate_source(_request(tmp_path))

    assert source.run_root == latest.resolve()
    assert source.metadata["strategy"] == "sag2"
    assert [path.name for path in source.source_files] == [
        "search_results.json",
        "qa_results.json",
    ]


def test_locate_source_honors_explicit_run_id(tmp_path):
    rows = [{"question_index": 1, "question": "Q", "retrieved_docs": ["C"]}]
    qa_rows = [{"question_index": 1, "question": "Q", "predicted_answer": "A"}]
    _write_sag_run(tmp_path, "20260101_000001", rows, qa_rows)
    selected = _write_sag_run(tmp_path, "20260102_000001", rows, qa_rows)

    source = SAGAdapter().locate_source(replace(_request(tmp_path), source_run_id=selected.name))

    assert source.run_root == selected.resolve()


def test_locate_source_requires_qa_result(tmp_path):
    run_root = tmp_path / "musique" / "sag2" / "20260101_000001"
    run_root.mkdir(parents=True)
    (run_root / "search_results.json").write_text("[]", encoding="utf-8")

    with pytest.raises(SourceRunNotFoundError, match="QA result"):
        SAGAdapter().locate_source(_request(tmp_path))


def test_conversion_joins_search_contexts_and_qa_answers(tmp_path):
    rows = [{"question_index": 1, "question": "Q", "retrieved_docs": ["C1", "C2"]}]
    qa_rows = [{"question_index": 1, "question": "Q", "predicted_answer": "A"}]
    _write_sag_run(tmp_path, "20260101_000001", rows, qa_rows)

    source = SAGAdapter().locate_source(_request(tmp_path))
    conversion = SAGAdapter().convert(source)

    assert conversion.rows == (
        {
            "canonical_row_id": 0,
            "question": "Q",
            "contexts": ["C1", "C2"],
            "generated_answer": "A",
        },
    )


def test_conversion_uses_question_index_to_preserve_duplicate_questions(tmp_path):
    source_root = tmp_path / "output"
    rows = [
        {"question_index": 1, "question": "duplicate", "retrieved_docs": ["C1"]},
        {"question_index": 2, "question": "duplicate", "retrieved_docs": ["C2"]},
    ]
    qa_rows = [
        {"question_index": 1, "question": "duplicate", "predicted_answer": "A1"},
        {"question_index": 2, "question": "duplicate", "predicted_answer": "A2"},
    ]
    _write_sag_run(source_root, "20260101_000001", rows, qa_rows)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "musique.json").write_text(
        json.dumps(
            [
                {"id": "sample-a", "question": "duplicate", "answer": "A1", "paragraphs": []},
                {"id": "sample-b", "question": "duplicate", "answer": "A2", "paragraphs": []},
            ]
        ),
        encoding="utf-8",
    )

    registry = AdapterRegistry()
    registry.register(SAGAdapter())
    service = PredictionConversionService(
        registry=registry,
        ground_truth_repository=GroundTruthRepository(dataset_dir),
        resolver=ArtifactLayoutResolver(),
        store=JudgeArtifactStore(),
    )
    result = service.convert(
        ConversionRequest(
            project="sag",
            dataset="musique",
            input_root=source_root,
            dataset_dir=dataset_dir,
            artifact_run_root=tmp_path / "artifacts",
        )
    )

    predictions = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert [row["id"] for row in predictions] == [0, 1]
    assert [row["contexts"] for row in predictions] == [["C1"], ["C2"]]


def test_locate_source_rejects_non_sag2_strategy_even_with_explicit_input_root(tmp_path):
    rows = [{"question_index": 1, "question": "Q", "retrieved_docs": ["C"]}]
    qa_rows = [{"question_index": 1, "question": "Q", "predicted_answer": "A"}]
    vector_root = tmp_path / "musique" / "vector"
    run_root = vector_root / "20260101_000001"
    qa_root = run_root / "qa_20260101_010101"
    qa_root.mkdir(parents=True)
    (run_root / "search_results.json").write_text(json.dumps(rows), encoding="utf-8")
    (qa_root / "qa_results.json").write_text(json.dumps({"per_example": qa_rows}), encoding="utf-8")

    with pytest.raises(SourceRunNotFoundError, match="SAG2 search result"):
        SAGAdapter().locate_source(replace(_request(tmp_path), input_root=vector_root))
