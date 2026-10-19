import json

import pytest

from pipeline.evaluation.judge.adapters.graphrag import GraphRAGAdapter
from pipeline.evaluation.judge.errors import SourceRunNotFoundError
from pipeline.evaluation.judge.models import ConversionRequest, SourceRun
from scripts.run_llm_judge import build_parser, cmd_convert


def _write_canonical(run_dir, dataset):
    (run_dir / "qa").mkdir(parents=True, exist_ok=True)
    (run_dir / "response").mkdir(parents=True, exist_ok=True)
    (run_dir / "qa" / "qa_results.json").write_text(
        json.dumps({"per_example": []}), encoding="utf-8"
    )
    (run_dir / "response" / f"graphrag_{dataset}_result.json").write_text(
        json.dumps([]), encoding="utf-8"
    )


def _make_request(input_root, dataset, artifact_root):
    return ConversionRequest(
        project="graphrag",
        dataset=dataset,
        input_root=input_root,
        dataset_dir=artifact_root / "datasets",
        artifact_run_root=artifact_root,
    )


def test_graphrag_adapter_joins_one_based_qa_to_zero_based_retrieval(tmp_path):
    run = tmp_path
    (run / "qa").mkdir()
    (run / "response").mkdir()
    (run / "qa" / "qa_results.json").write_text(
        json.dumps(
            {
                "per_example": [
                    {
                        "dataset_sample_id": "id-A",
                        "question_index": 1,
                        "question": "question A",
                        "predicted_answer": "answer A",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "response" / "graphrag_test_result.json").write_text(
        json.dumps(
            [
                {
                    "dataset_sample_id": "id-A",
                    "question_index": 0,
                    "question": "question A",
                    "retrieved_docs": ["correct context"],
                }
            ]
        ),
        encoding="utf-8",
    )

    source = SourceRun(
        project="graphrag",
        dataset="test",
        run_root=run.resolve(),
        artifact_run_root=run.resolve(),
    )
    rows = list(GraphRAGAdapter().convert(source).rows)

    assert rows[0]["context"] == "correct context"


def test_locate_source_uses_batch_dir_as_source_run_id(tmp_path):
    input_root = tmp_path / "outputs"
    batch = input_root / "test" / "create_final_entities_20240101"
    _write_canonical(batch, "test")

    source = GraphRAGAdapter().locate_source(_make_request(input_root, "test", tmp_path))

    assert source.metadata["source_run_id"] == "create_final_entities_20240101"


def test_locate_source_flat_layout_falls_back_to_flat(tmp_path):
    input_root = tmp_path / "outputs"
    _write_canonical(input_root / "test", "test")

    source = GraphRAGAdapter().locate_source(_make_request(input_root, "test", tmp_path))

    assert source.metadata["source_run_id"] == "flat"


def test_locate_source_auto_selects_latest_batch(tmp_path):
    input_root = tmp_path / "outputs"
    _write_canonical(input_root / "test" / "batch-a", "test")
    _write_canonical(input_root / "test" / "batch-b", "test")

    source = GraphRAGAdapter().locate_source(_make_request(input_root, "test", tmp_path))

    assert source.run_root.name == "batch-b"
    assert source.metadata["source_run_id"] == "batch-b"


def test_locate_source_selects_explicit_batch(tmp_path):
    input_root = tmp_path / "outputs"
    _write_canonical(input_root / "test" / "batch-a", "test")
    _write_canonical(input_root / "test" / "batch-b", "test")

    request = _make_request(input_root, "test", tmp_path)
    request = ConversionRequest(
        project=request.project,
        dataset=request.dataset,
        input_root=request.input_root,
        dataset_dir=request.dataset_dir,
        artifact_run_root=request.artifact_run_root,
        source_run_id="batch-a",
    )
    source = GraphRAGAdapter().locate_source(request)

    assert source.run_root.name == "batch-a"
    assert source.metadata["source_run_id"] == "batch-a"


def test_locate_source_rejects_unknown_explicit_batch(tmp_path):
    input_root = tmp_path / "outputs"
    _write_canonical(input_root / "test" / "batch-a", "test")
    request = _make_request(input_root, "test", tmp_path)
    request = ConversionRequest(
        project=request.project,
        dataset=request.dataset,
        input_root=request.input_root,
        dataset_dir=request.dataset_dir,
        artifact_run_root=request.artifact_run_root,
        source_run_id="missing",
    )

    with pytest.raises(SourceRunNotFoundError, match="source run not found"):
        GraphRAGAdapter().locate_source(request)


def test_graphrag_dry_run_does_not_require_response_or_write(tmp_path, capsys):
    input_root = tmp_path / "outputs"
    batch = input_root / "test" / "batch-001"
    (batch / "qa").mkdir(parents=True)
    (batch / "qa" / "qa_results.json").write_text(json.dumps({"per_example": []}), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    args = build_parser().parse_args(
        [
            "convert",
            "--project",
            "graphrag",
            "--input-root",
            str(input_root),
            "--datasets",
            "test",
            "--artifact-run-root",
            str(artifact_root),
            "--dataset-dir",
            str(tmp_path / "not-needed-in-dry-run"),
            "--dry-run",
        ]
    )

    assert cmd_convert(args) == 0
    assert not artifact_root.exists()
    out = capsys.readouterr().out
    assert str(batch / "qa" / "qa_results.json") in out
    assert "graphrag_test_result.json" not in out
    assert "no directories created, no files written" in out
