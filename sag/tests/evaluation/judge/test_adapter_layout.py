"""Tests for HippoRAG2, HyperGraphRAG, HyperRAG, LightRAG, GraphRAG adapters.

Covers locate_source for all supported shapes and convert for exact files.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.evaluation.judge.adapters.graphrag import GraphRAGAdapter
from pipeline.evaluation.judge.adapters.hipporag2 import HippoRAG2Adapter
from pipeline.evaluation.judge.adapters.hypergraphrag import HyperGraphRAGAdapter
from pipeline.evaluation.judge.adapters.hyperrag import HyperRAGAdapter
from pipeline.evaluation.judge.adapters.lightrag import LightRAGAdapter
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    SourceRunAmbiguousError,
    SourceRunNotFoundError,
)
from pipeline.evaluation.judge.models import ConversionRequest, SourceRun


def _request(input_root: Path, dataset: str = "test"):
    return ConversionRequest(
        project="test",
        dataset=dataset,
        input_root=input_root,
        dataset_dir=input_root,
        artifact_run_root=input_root,
    )


# ---------------------------------------------------------------------------
# HippoRAG2 — four supported shapes + ambiguity
# ---------------------------------------------------------------------------


class TestHippoRAG2Adapter:
    def test_locate_source_outputs_root(self, tmp_path):
        """Shape 1: outputs root — input_root/<dataset>/<llm>_<emb>/qa_result"""
        ds = tmp_path / "test" / "gpt-4_text-emb" / "qa_result"
        ds.mkdir(parents=True)
        (ds / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        adapter = HippoRAG2Adapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hipporag2"
        assert source.metadata["source_run_id"] == "gpt-4_text-emb"

    def test_source_run_id_selects_the_requested_model_directory(self, tmp_path):
        for model in ("gpt-4_emb-a", "gpt-4_emb-b"):
            qa = tmp_path / "test" / model / "qa_result"
            qa.mkdir(parents=True)
            (qa / "qa_results_latest.json").write_text(json.dumps({"results": []}))

        source = HippoRAG2Adapter().locate_source(
            replace(_request(tmp_path), source_run_id="gpt-4_emb-b")
        )

        assert source.run_root.parent.name == "gpt-4_emb-b"
        assert source.metadata["source_run_id"] == "gpt-4_emb-b"

    def test_unknown_source_run_id_raises(self, tmp_path):
        qa = tmp_path / "test" / "gpt-4_emb-a" / "qa_result"
        qa.mkdir(parents=True)
        (qa / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        with pytest.raises(SourceRunNotFoundError, match="source run not found"):
            HippoRAG2Adapter().locate_source(replace(_request(tmp_path), source_run_id="missing"))

    def test_locate_source_dataset_root(self, tmp_path):
        """Shape 2: dataset root — input_root/<dataset>/qa_result"""
        qa = tmp_path / "test" / "qa_result"
        qa.mkdir(parents=True)
        (qa / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        adapter = HippoRAG2Adapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hipporag2"

    def test_locate_source_exact_model_root(self, tmp_path):
        """Shape 3: exact model root — input_root/<llm>_<emb>/qa_result"""
        qa = tmp_path / "gpt-4_text-emb" / "qa_result"
        qa.mkdir(parents=True)
        (qa / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        adapter = HippoRAG2Adapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hipporag2"
        assert source.run_root.name == "qa_result"

    def test_locate_source_exact_qa_result_root(self, tmp_path):
        """Shape 4: exact qa_result root — input_root/qa_results_latest.json"""
        (tmp_path / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        adapter = HippoRAG2Adapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hipporag2"

    def test_ambiguity_raises(self, tmp_path):
        """Multiple model dirs -> SourceRunAmbiguousError."""
        qa1 = tmp_path / "gpt-4_emb1" / "qa_result"
        qa1.mkdir(parents=True)
        (qa1 / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        qa2 = tmp_path / "gpt-4_emb2" / "qa_result"
        qa2.mkdir(parents=True)
        (qa2 / "qa_results_latest.json").write_text(json.dumps({"results": []}))
        adapter = HippoRAG2Adapter()
        with pytest.raises(SourceRunAmbiguousError):
            adapter.locate_source(_request(tmp_path))

    def test_missing_raises(self, tmp_path):
        adapter = HippoRAG2Adapter()
        with pytest.raises(SourceRunNotFoundError):
            adapter.locate_source(_request(tmp_path))

    def test_legacy_caches_root_is_rejected(self, tmp_path):
        legacy_root = tmp_path / "caches"
        legacy_root.mkdir()
        with pytest.raises(SourceRunNotFoundError, match="outputs"):
            HippoRAG2Adapter().locate_source(_request(legacy_root))

    def test_convert(self, tmp_path):
        qa = tmp_path / "qa_results_latest.json"
        qa.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "dataset_sample_id": "sample-1",
                            "question": "Q",
                            "answer": "A",
                            "gold_answers": ["G"],
                            "docs": ["D1"],
                        }
                    ]
                }
            )
        )
        adapter = HippoRAG2Adapter()
        source = SourceRun(
            project="hipporag2",
            dataset="test",
            run_root=tmp_path.resolve(),
            artifact_run_root=tmp_path.resolve(),
        )
        conv = adapter.convert(source)
        assert len(conv.rows) == 1
        assert conv.rows[0]["question"] == "Q"
        assert conv.rows[0]["dataset_sample_id"] == "sample-1"

    def test_graphrag_legacy_caches_root_is_rejected(self, tmp_path):
        legacy_root = tmp_path / "caches"
        legacy_root.mkdir()
        with pytest.raises(SourceRunNotFoundError, match="outputs"):
            GraphRAGAdapter().locate_source(_request(legacy_root))


# ---------------------------------------------------------------------------
# HyperGraphRAG
# ---------------------------------------------------------------------------


class TestHyperGraphRAGAdapter:
    def test_locate_source(self, tmp_path):
        ds = tmp_path / "test" / "response"
        ds.mkdir(parents=True)
        (ds / "hybrid_test_result.json").write_text(json.dumps([]))
        adapter = HyperGraphRAGAdapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hypergraphrag"

    def test_locate_source_selects_explicit_response_run(self, tmp_path):
        for run_id in ("20260101_000001", "20260102_000001"):
            response = tmp_path / "test" / "response" / run_id
            response.mkdir(parents=True)
            (response / "hybrid_test_result.json").write_text(json.dumps([]))
        source = HyperGraphRAGAdapter().locate_source(
            replace(_request(tmp_path), source_run_id="20260101_000001")
        )
        assert source.metadata["source_run_id"] == "20260101_000001"
        assert source.source_files[0].parent.name == "20260101_000001"

    def test_locate_source_auto_selects_latest_response_run(self, tmp_path):
        for run_id in ("20260101_000001", "20260102_000001"):
            response = tmp_path / "test" / "response" / run_id
            response.mkdir(parents=True)
            (response / "hybrid_test_result.json").write_text(json.dumps([]))
        source = HyperGraphRAGAdapter().locate_source(_request(tmp_path))
        assert source.metadata["source_run_id"] == "20260102_000001"
        assert source.source_files[0].parent.name == "20260102_000001"

    def test_response_directory_without_a_result_file_raises(self, tmp_path):
        (tmp_path / "test" / "response").mkdir(parents=True)
        with pytest.raises(SourceRunNotFoundError, match="no hybrid result"):
            HyperGraphRAGAdapter().locate_source(_request(tmp_path))

    def test_locate_source_direct_file(self, tmp_path):
        """Support direct result file in dataset dir."""
        ds = tmp_path / "test"
        ds.mkdir(parents=True)
        (ds / "hybrid_test_result.json").write_text(json.dumps([]))
        adapter = HyperGraphRAGAdapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "hypergraphrag"

    def test_missing_raises(self, tmp_path):
        adapter = HyperGraphRAGAdapter()
        with pytest.raises(SourceRunNotFoundError):
            adapter.locate_source(_request(tmp_path))

    def test_convert(self, tmp_path):
        resp = tmp_path / "response"
        resp.mkdir()
        (resp / "hybrid_test_result.json").write_text(
            json.dumps([{"id": "sample-1", "query": "Q", "context": "ctx", "pred_answer": "A"}])
        )
        adapter = HyperGraphRAGAdapter()
        source = SourceRun(
            project="hypergraphrag",
            dataset="test",
            run_root=tmp_path.resolve(),
            artifact_run_root=tmp_path.resolve(),
        )
        conv = adapter.convert(source)
        assert len(conv.rows) == 1
        assert conv.rows[0]["dataset_sample_id"] == "sample-1"


# ---------------------------------------------------------------------------
# HyperRAG — dataset root, response root, exact JSON, multi-mode rejection
# ---------------------------------------------------------------------------


class TestHyperRAGAdapter:
    def test_locate_source_dataset_root(self, tmp_path):
        """Shape: dataset root — input_root/<dataset>/response/<mode>_<ds>_result.json"""
        ds = tmp_path / "test" / "response"
        ds.mkdir(parents=True)
        (ds / "hyper_test_result.json").write_text(json.dumps([]))
        adapter = HyperRAGAdapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.metadata["mode"] == "hyper"

    def test_source_run_id_is_the_selected_response_directory(self, tmp_path):
        for run_id in ("20260101_000001", "20260102_000001"):
            response = tmp_path / "test" / "response" / run_id
            response.mkdir(parents=True)
            (response / "hyper_test_result.json").write_text(json.dumps([]))
        source = HyperRAGAdapter().locate_source(
            replace(
                _request(tmp_path),
                mode="hyper",
                source_run_id="20260101_000001",
            )
        )
        assert source.metadata["source_run_id"] == "20260101_000001"
        assert source.run_root.name == "20260101_000001"

    def test_auto_selects_latest_timestamped_response_run(self, tmp_path):
        for run_id in ("20260101_000001", "20260102_000001"):
            response = tmp_path / "test" / "response" / run_id
            response.mkdir(parents=True)
            (response / "hyper_test_result.json").write_text(json.dumps([]))
        source = HyperRAGAdapter().locate_source(replace(_request(tmp_path), mode="hyper"))
        assert source.metadata["source_run_id"] == "20260102_000001"
        assert source.run_root.name == "20260102_000001"

    def test_unknown_source_run_id_does_not_fall_back_to_flat_file(self, tmp_path):
        response = tmp_path / "test" / "response"
        response.mkdir(parents=True)
        (response / "hyper_test_result.json").write_text(json.dumps([]))
        with pytest.raises(SourceRunNotFoundError):
            HyperRAGAdapter().locate_source(
                replace(
                    _request(tmp_path),
                    mode="hyper",
                    source_run_id="20260101_000001",
                )
            )

    def test_locate_source_response_root(self, tmp_path):
        """Shape: response root — input_root/response/<mode>_<ds>_result.json"""
        resp = tmp_path / "response"
        resp.mkdir(parents=True)
        (resp / "hyper_test_result.json").write_text(json.dumps([]))
        adapter = HyperRAGAdapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.metadata["mode"] == "hyper"

    def test_locate_source_exact_json(self, tmp_path):
        """Shape: exact JSON — input_root is a .json file"""
        json_file = tmp_path / "hyper_test_result.json"
        json_file.write_text(json.dumps([]))
        adapter = HyperRAGAdapter()
        source = adapter.locate_source(
            ConversionRequest(
                project="test",
                dataset="test",
                input_root=json_file,
                dataset_dir=tmp_path,
                artifact_run_root=tmp_path,
            )
        )
        assert source.metadata["mode"] == "hyper"
        assert len(source.source_files) == 1
        assert source.source_files[0] == json_file

    def test_locate_source_exact_json_hyper_lite(self, tmp_path):
        """Exact JSON with hyper-lite mode"""
        json_file = tmp_path / "hyper-lite_test_result.json"
        json_file.write_text(json.dumps([]))
        adapter = HyperRAGAdapter()
        source = adapter.locate_source(
            ConversionRequest(
                project="test",
                dataset="test",
                input_root=json_file,
                dataset_dir=tmp_path,
                artifact_run_root=tmp_path,
            )
        )
        assert source.metadata["mode"] == "hyper-lite"

    def test_multi_mode_raises(self, tmp_path):
        """Multiple modes in response dir -> AdapterConversionError"""
        resp = tmp_path / "test" / "response"
        resp.mkdir(parents=True)
        (resp / "hyper_test_result.json").write_text(json.dumps([]))
        (resp / "naive_test_result.json").write_text(json.dumps([]))
        adapter = HyperRAGAdapter()
        with pytest.raises(AdapterConversionError):
            adapter.locate_source(_request(tmp_path))

    def test_missing_raises(self, tmp_path):
        adapter = HyperRAGAdapter()
        with pytest.raises(SourceRunNotFoundError):
            adapter.locate_source(_request(tmp_path))

    def test_convert_uses_source_file(self, tmp_path):
        """convert() uses the exact source file, not reconstructing path."""
        resp = tmp_path / "response"
        resp.mkdir()
        resp_file = resp / "hyper_test_result.json"
        resp_file.write_text(
            json.dumps([{"id": "sample-1", "query": "Q", "context": "ctx", "pred_answer": "A"}])
        )
        adapter = HyperRAGAdapter()
        source = SourceRun(
            project="hyperrag",
            dataset="test",
            run_root=tmp_path.resolve(),
            artifact_run_root=tmp_path.resolve(),
            source_files=(resp_file,),
            metadata={"mode": "hyper"},
        )
        conv = adapter.convert(source)
        assert len(conv.rows) == 1
        assert conv.rows[0]["question"] == "Q"
        assert conv.rows[0]["dataset_sample_id"] == "sample-1"

    def test_convert_exact_json_file(self, tmp_path):
        """convert() with exact JSON as source file"""
        json_file = tmp_path / "naive_test_result.json"
        json_file.write_text(
            json.dumps([{"id": "sample-1", "query": "Q", "context": "ctx", "pred_answer": "A"}])
        )
        adapter = HyperRAGAdapter()
        source = SourceRun(
            project="hyperrag",
            dataset="test",
            run_root=tmp_path.resolve(),
            artifact_run_root=tmp_path.resolve(),
            source_files=(json_file,),
            metadata={"mode": "naive"},
        )
        conv = adapter.convert(source)
        assert len(conv.rows) == 1


# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------


class TestLightRAGAdapter:
    def test_locate_source(self, tmp_path):
        ds = tmp_path / "test" / "response"
        ds.mkdir(parents=True)
        (ds / "hybrid_test_result.json").write_text(json.dumps([]))
        adapter = LightRAGAdapter()
        source = adapter.locate_source(_request(tmp_path))
        assert source.project == "lightrag"

    def test_source_run_id_tracks_the_auto_selected_response_run(self, tmp_path):
        for run_id in ("20260101_000001", "20260102_000001"):
            response = tmp_path / "test" / "response" / run_id
            response.mkdir(parents=True)
            (response / "hybrid_test_result.json").write_text(json.dumps([]))
        source = LightRAGAdapter().locate_source(_request(tmp_path))
        assert source.metadata["source_run_id"] == "20260102_000001"
        assert source.source_files[0].parent.name == "20260102_000001"

    def test_missing_raises(self, tmp_path):
        adapter = LightRAGAdapter()
        with pytest.raises(SourceRunNotFoundError):
            adapter.locate_source(_request(tmp_path))

    def test_convert(self, tmp_path):
        resp = tmp_path / "response"
        resp.mkdir()
        (resp / "hybrid_test_result.json").write_text(
            json.dumps(
                [{"id": "sample-1", "query": "Q", "retrieved_docs": ["D1"], "pred_answer": "A"}]
            )
        )
        adapter = LightRAGAdapter()
        source = SourceRun(
            project="lightrag",
            dataset="test",
            run_root=tmp_path.resolve(),
            artifact_run_root=tmp_path.resolve(),
        )
        conv = adapter.convert(source)
        assert len(conv.rows) == 1
        assert conv.metadata["empty_retrieved_docs_count"] == 0


# ---------------------------------------------------------------------------
# GraphRAG adapter — alignment and empty question rejection
# ---------------------------------------------------------------------------


class TestGraphRAGAdapter:
    def test_question_text_wins_when_indices_offset(self, tmp_path):
        """QA and retrieval rows aligned by dataset_sample_id, not by question_index."""
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)

        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-001",
                            "question_index": 1,
                            "question": "Question one?",
                            "predicted_answer": "Answer one",
                            "gold_answers": ["Gold one"],
                        },
                        {
                            "dataset_sample_id": "id-002",
                            "question_index": 2,
                            "question": "Question two?",
                            "predicted_answer": "Answer two",
                            "gold_answers": ["Gold two"],
                        },
                    ]
                }
            )
        )
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_sample_id": "id-001",
                        "question_index": 0,
                        "question": "Question one?",
                        "retrieved_docs": ["Context one"],
                    },
                    {
                        "dataset_sample_id": "id-002",
                        "question_index": 1,
                        "question": "Question two?",
                        "retrieved_docs": ["Context two"],
                    },
                ]
            )
        )

        adapter = GraphRAGAdapter()
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        conversion = adapter.convert(source)
        assert [(r["question"], r["context"]) for r in conversion.rows] == [
            ("Question one?", "Context one"),
            ("Question two?", "Context two"),
        ]

    def test_empty_question_in_retrieval_raises(self, tmp_path):
        """Empty/whitespace question in retrieval must raise."""
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)

        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-001",
                            "question_index": 1,
                            "question": "Question one?",
                            "predicted_answer": "Answer one",
                            "gold_answers": ["Gold one"],
                        }
                    ]
                }
            )
        )
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_sample_id": "id-001",
                        "question_index": 0,
                        "question": " ",
                        "retrieved_docs": ["Wrong"],
                    }
                ]
            )
        )

        adapter = GraphRAGAdapter()
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        with pytest.raises(AdapterConversionError, match="missing non-empty"):
            adapter.convert(source)

    def test_empty_question_in_retrieval_raises_full_context(self, tmp_path):
        """Item 5: retrieval row missing dataset_sample_id must raise, not skip."""
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)

        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-001",
                            "question_index": 1,
                            "question": "Question one?",
                            "predicted_answer": "Answer one",
                            "gold_answers": ["Gold one"],
                        },
                        {
                            "dataset_sample_id": "id-002",
                            "question_index": 2,
                            "question": "Question two?",
                            "predicted_answer": "Answer two",
                            "gold_answers": ["Gold two"],
                        },
                    ]
                }
            )
        )
        # retrieval file with a row missing dataset_sample_id
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_sample_id": "id-001",
                        "question": "Question one?",
                        "retrieved_docs": ["Context one"],
                    },
                    {
                        "question": "",
                        "retrieved_docs": ["Skipped context"],
                    },
                    {
                        "dataset_sample_id": "id-002",
                        "question": "Question two?",
                        "retrieved_docs": ["Context two"],
                    },
                ]
            )
        )

        adapter = GraphRAGAdapter()
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        with pytest.raises(AdapterConversionError, match="missing a non-empty"):
            adapter.convert(source)

    def test_duplicate_question_in_retrieval_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)

        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-001",
                            "question_index": 1,
                            "question": "Q dup",
                            "predicted_answer": "A",
                            "gold_answers": ["G"],
                        }
                    ]
                }
            )
        )
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {"dataset_sample_id": "id-001", "question": "Q dup", "retrieved_docs": ["C1"]},
                    {"dataset_sample_id": "id-001", "question": "Q dup", "retrieved_docs": ["C2"]},
                ]
            )
        )

        adapter = GraphRAGAdapter()
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        with pytest.raises(AdapterConversionError, match="duplicate"):
            adapter.convert(source)

    def test_extra_retrieval_question_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)
        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-001",
                            "question": "Q1",
                            "predicted_answer": "A1",
                            "gold_answers": ["G1"],
                        }
                    ]
                }
            )
        )
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {"dataset_sample_id": "id-001", "question": "Q1", "retrieved_docs": ["C1"]},
                    {
                        "dataset_sample_id": "id-EXTRA",
                        "question": "EXTRA",
                        "retrieved_docs": ["C2"],
                    },
                ]
            )
        )
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        with pytest.raises(AdapterConversionError, match="retrieval-only"):
            GraphRAGAdapter().convert(source)

    def test_missing_context_for_qa_question_raises(self, tmp_path):
        """QA row with a dataset_sample_id not in retrieval must raise."""
        run_dir = tmp_path / "run"
        qa_dir = run_dir / "qa"
        resp_dir = run_dir / "response"
        qa_dir.mkdir(parents=True)
        resp_dir.mkdir(parents=True)

        (qa_dir / "qa_results.json").write_text(
            json.dumps(
                {
                    "per_example": [
                        {
                            "dataset_sample_id": "id-missing",
                            "question_index": 1,
                            "question": "Missing context Q?",
                            "predicted_answer": "A",
                            "gold_answers": ["G"],
                        }
                    ]
                }
            )
        )
        (resp_dir / "graphrag_test_result.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_sample_id": "id-other",
                        "question": "Other Q?",
                        "retrieved_docs": ["Context"],
                    }
                ]
            )
        )

        adapter = GraphRAGAdapter()
        source = SourceRun(
            project="graphrag",
            dataset="test",
            run_root=run_dir.resolve(),
            artifact_run_root=run_dir.resolve(),
        )
        with pytest.raises(AdapterConversionError, match="no retrieval context"):
            adapter.convert(source)
