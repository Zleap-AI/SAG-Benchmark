"""Regression tests for GraphRAG question-text alignment.

Tests that 1-based QA indices and 0-based retrieval indices are aligned by
question text, not by position; and that missing questions raise errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.evaluation.judge.adapters.graphrag import GraphRAGAdapter
from pipeline.evaluation.judge.errors import AdapterConversionError
from pipeline.evaluation.judge.models import SourceRun


def _make_source(run_dir: Path, dataset: str = "test") -> SourceRun:
    return SourceRun(
        project="graphrag",
        dataset=dataset,
        run_root=run_dir.resolve(),
        artifact_run_root=run_dir.resolve(),
    )


def _write_canonical_run(run_dir: Path, retrieval_rows: list[dict]) -> None:
    qa_dir = run_dir / "qa"
    response_dir = run_dir / "response"
    qa_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)

    qa_rows = [
        {
            "question_index": 1,
            "dataset_sample_id": "sample-1",
            "question": "Question one?",
            "predicted_answer": "Answer one",
            "gold_answers": ["Gold one"],
        },
        {
            "question_index": 2,
            "dataset_sample_id": "sample-2",
            "question": "Question two?",
            "predicted_answer": "Answer two",
            "gold_answers": ["Gold two"],
        },
    ]
    (qa_dir / "qa_results.json").write_text(json.dumps({"per_example": qa_rows}), encoding="utf-8")
    (response_dir / "graphrag_test_result.json").write_text(
        json.dumps(retrieval_rows), encoding="utf-8"
    )


def test_question_text_wins_when_indices_are_deliberately_offset(
    tmp_path: Path,
) -> None:
    """QA indices are 1-based, retrieval indices are 0-based — match by text."""
    _write_canonical_run(
        tmp_path,
        [
            {
                "question_index": 0,
                "dataset_sample_id": "sample-1",
                "question": "Question one?",
                "retrieved_docs": ["Context one"],
            },
            {
                "question_index": 1,
                "dataset_sample_id": "sample-2",
                "question": "Question two?",
                "retrieved_docs": ["Context two"],
            },
        ],
    )

    adapter = GraphRAGAdapter()
    conversion = adapter.convert(_make_source(tmp_path))

    assert [(r["question"], r["context"]) for r in conversion.rows] == [
        ("Question one?", "Context one"),
        ("Question two?", "Context two"),
    ]


def test_missing_retrieval_question_fails_instead_of_using_index(
    tmp_path: Path,
) -> None:
    """Empty/whitespace question in retrieval must raise, not fall back to index."""
    _write_canonical_run(
        tmp_path,
        [
            {
                "question_index": 0,
                "dataset_sample_id": "sample-1",
                "question": " ",
                "retrieved_docs": ["Wrong"],
            }
        ],
    )

    adapter = GraphRAGAdapter()
    with pytest.raises(AdapterConversionError, match="missing non-empty 'question'"):
        adapter.convert(_make_source(tmp_path))
