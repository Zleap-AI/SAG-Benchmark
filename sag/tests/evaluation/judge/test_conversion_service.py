"""Tests for PredictionConversionService — manifest, hashing, overwrite, dry-run."""

import json
from pathlib import Path

import pytest

from pipeline.evaluation.judge.adapters.registry import AdapterRegistry
from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
)
from pipeline.evaluation.judge.conversion import PredictionConversionService
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    AdapterNotFoundError,
    PredictionValidationError,
)
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)


class _SimpleAdapter:
    name = "simple"

    def locate_source(self, request):
        return SourceRun(
            project="simple",
            dataset=request.dataset,
            run_root=request.input_root.resolve(),
            artifact_run_root=request.input_root.resolve(),
            metadata={"source_run_id": request.source_run_id or "latest"},
        )

    def convert(self, source):
        return AdapterConversion(
            rows=(
                {
                    "dataset_sample_id": "sample/question_1.json",
                    "question": "What is X?",
                    "context": "X is a letter.",
                    "contexts": ["X is a letter."],
                    "generated_answer": "X",
                },
            ),
            metadata={"adapter_version": "1.0.0"},
        )


def _make_service(dataset_dir: Path):
    registry = AdapterRegistry()
    registry.register(_SimpleAdapter())
    gt_repo = GroundTruthRepository(dataset_dir)
    resolver = ArtifactLayoutResolver()
    store = JudgeArtifactStore()
    return PredictionConversionService(
        registry=registry,
        ground_truth_repository=gt_repo,
        resolver=resolver,
        store=store,
    )


def _write_dataset(dataset_dir: Path, dataset_name: str):
    ds_file = dataset_dir / f"{dataset_name}.json"
    ds_file.write_text(
        json.dumps(
            [
                {
                    "id": "sample/question_1.json",
                    "question": "What is X?",
                    "answer": "X",
                    "paragraphs": [
                        {
                            "text": "X is a letter.",
                            "is_supporting": True,
                        }
                    ],
                }
            ]
        )
    )


class TestConversionService:
    def test_basic_convert(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"

        _write_dataset(dataset_dir, "sample")

        service = _make_service(dataset_dir)
        request = ConversionRequest(
            project="simple",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
        )
        result = service.convert(request)

        assert result.row_count == 1
        assert result.predictions_path.exists()
        assert result.manifest_path.exists()

    def test_manifest_content(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"

        _write_dataset(dataset_dir, "sample")

        service = _make_service(dataset_dir)
        request = ConversionRequest(
            project="simple",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
        )
        result = service.convert(request)

        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["project"] == "simple"
        assert manifest["dataset"] == "sample"
        assert manifest["row_count"] == 1
        assert manifest["schema_version"] == 1

    def test_allow_overwrite_false_rejects(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"

        _write_dataset(dataset_dir, "sample")

        service = _make_service(dataset_dir)
        request = ConversionRequest(
            project="simple",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
            allow_overwrite=True,
        )
        result = service.convert(request)
        assert result.row_count == 1

        # Second call with allow_overwrite=False should fail
        request2 = ConversionRequest(
            project="simple",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
            allow_overwrite=False,
        )
        with pytest.raises(PredictionValidationError, match="already exists"):
            service.convert(request2)

    def test_unknown_project_raises(self, tmp_path):
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        service = _make_service(dataset_dir)
        request = ConversionRequest(
            project="nonexistent",
            dataset="sample",
            input_root=tmp_path,
            dataset_dir=dataset_dir,
            artifact_run_root=tmp_path,
        )
        with pytest.raises(AdapterNotFoundError):
            service.convert(request)

    def test_input_digest_present(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"

        _write_dataset(dataset_dir, "sample")

        service = _make_service(dataset_dir)
        request = ConversionRequest(
            project="simple",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
        )
        result = service.convert(request)
        assert len(result.input_digest) == 64  # SHA-256 hex
        assert result.output_digest

    def test_duplicate_question_rows_are_rejected(self, tmp_path):
        """Two source rows must not map to the same canonical dataset row."""
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"
        _write_dataset(dataset_dir, "sample")

        class _DupAdapter(_SimpleAdapter):
            name = "dup"

            def convert(self, source):
                return AdapterConversion(
                    rows=(
                        {
                            "dataset_sample_id": "sample/question_1.json",
                            "question": "What is X?",
                            "context": "c1",
                            "generated_answer": "A",
                        },
                        {
                            "dataset_sample_id": "sample/question_1.json",
                            "question": "What is X?",
                            "context": "c2",
                            "generated_answer": "A",
                        },
                    ),
                    metadata={"adapter_version": "1.0.0"},
                )

        registry = AdapterRegistry()
        registry.register(_DupAdapter())
        service = PredictionConversionService(
            registry=registry,
            ground_truth_repository=GroundTruthRepository(dataset_dir),
            resolver=ArtifactLayoutResolver(),
            store=JudgeArtifactStore(),
        )
        request = ConversionRequest(
            project="dup",
            dataset="sample",
            input_root=input_root,
            dataset_dir=dataset_dir,
            artifact_run_root=artifact_root,
            allow_overwrite=True,
        )
        with pytest.raises(AdapterConversionError, match="same dataset row 0"):
            service.convert(request)

    def test_dataset_sample_ids_keep_duplicate_questions_as_distinct_rows(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        artifact_root = tmp_path / "artifacts"
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
        (dataset_dir / "musique.json").write_text(json.dumps(rows), encoding="utf-8")

        class _IdAdapter(_SimpleAdapter):
            name = "id-adapter"

            def convert(self, source):
                return AdapterConversion(
                    rows=(
                        {
                            "dataset_sample_id": "sample-a",
                            "question": "duplicate question",
                            "context": "c1",
                            "generated_answer": "A",
                        },
                        {
                            "dataset_sample_id": "sample-b",
                            "question": "duplicate question",
                            "context": "c2",
                            "generated_answer": "B",
                        },
                    ),
                    metadata={"adapter_version": "1.0.0"},
                )

        registry = AdapterRegistry()
        registry.register(_IdAdapter())
        service = PredictionConversionService(
            registry=registry,
            ground_truth_repository=GroundTruthRepository(dataset_dir),
            resolver=ArtifactLayoutResolver(),
            store=JudgeArtifactStore(),
        )
        result = service.convert(
            ConversionRequest(
                project="id-adapter",
                dataset="musique",
                input_root=input_root,
                dataset_dir=dataset_dir,
                artifact_run_root=artifact_root,
                allow_overwrite=True,
            )
        )

        predictions = json.loads(result.predictions_path.read_text())
        assert [row["id"] for row in predictions] == [0, 1]
        assert [row["ground_truth"] for row in predictions] == ["A", "B"]
