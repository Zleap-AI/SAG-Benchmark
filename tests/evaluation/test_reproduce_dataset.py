"""Tests for method-local dataset cache export."""

import json

from pipeline.evaluation.utils.load_utils import DatasetLoader
from pipeline.evaluation.utils.reproduce_dataset import ReproduceDatasetExporter


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_export_writes_normalized_artifacts_manifest_and_legacy_sidecars(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    _write_json(source / "demo_corpus.json", [{"title": "T", "text": "body"}])
    _write_json(source / "demo.json", [{"id": "1", "question": "Q?", "answer": "A"}])

    manifest = ReproduceDatasetExporter(DatasetLoader("demo", source), tmp_path / "caches").export(
        legacy_sidecars=True, limit_documents=1, limit_questions=1
    )

    cache = tmp_path / "caches" / "demo"
    assert manifest["counts"] == {"documents": 1, "questions": 1}
    assert manifest["limits"] == {"documents": 1, "questions": 1}
    assert json.loads(
        (cache / "contexts" / "demo_corpus_docs.json").read_text(encoding="utf-8")
    ) == ["T\nbody"]
    questions = json.loads(
        (cache / "questions" / "demo_questions.json").read_text(encoding="utf-8")
    )
    assert questions[0]["id"] == "1"
    assert questions[0]["gold_answers"] == ["A"]
    assert json.loads((cache / "questions" / "demo_stage.json").read_text(encoding="utf-8")) == [
        "Q?"
    ]
    saved_manifest = json.loads((cache / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["source"]["dataset_root"] == str(source.resolve())
    assert len(saved_manifest["source"]["corpus_sha256"]) == 64
