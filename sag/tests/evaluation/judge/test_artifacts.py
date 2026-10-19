"""Tests for Judge artifact layout, path safety, sanitization, and store."""

import json
from pathlib import Path

import pytest

from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
    sanitize_path_component,
    sha256_file,
    sha256_json,
)
from pipeline.evaluation.judge.errors import ArtifactPathError


class TestSanitizePathComponent:
    def test_normal_name(self):
        assert sanitize_path_component("hello") == "hello"

    def test_special_chars_replaced(self):
        assert sanitize_path_component("my/model:v1") == "my_model_v1"

    def test_dotdot_rejected(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component("..")

    def test_dot_rejected(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component(".")

    def test_absolute_rejected(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component("/etc/passwd")

    def test_empty_rejected(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component("")

    def test_all_special_rejected(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component("///...")


class TestPathEscape:
    def test_predictions_under_evaluation(self):
        root = Path("/tmp/test_root")
        layout = ArtifactLayoutResolver.predictions(root, "my_dataset")
        assert "evaluation" in layout.predictions_file.parts
        assert "my_dataset" in str(layout.predictions_file)

    def test_dotdot_component_raises(self):
        with pytest.raises(ArtifactPathError):
            sanitize_path_component("../escape")


class TestInferRunRoot:
    def test_valid_predictions_path(self, tmp_path):
        pred_dir = tmp_path / "evaluation" / "predictions"
        pred_dir.mkdir(parents=True)
        pred_file = pred_dir / "predictions_hotpotqa.json"
        pred_file.write_text("[]")
        root = ArtifactLayoutResolver.infer_run_root(pred_file)
        assert root == tmp_path

    def test_three_layer_predictions_path(self, tmp_path):
        pred_dir = tmp_path / "evaluation" / "hyperrag" / "narrativeqa" / "run-001" / "predictions"
        pred_dir.mkdir(parents=True)
        pred_file = pred_dir / "predictions_narrativeqa.json"
        pred_file.write_text("[]")
        root = ArtifactLayoutResolver.infer_run_root(pred_file)
        assert root == tmp_path

    def test_three_layer_lineage(self, tmp_path):
        pred_dir = tmp_path / "evaluation" / "hyperrag" / "narrativeqa" / "run-001" / "predictions"
        pred_file = pred_dir / "predictions_narrativeqa.json"
        lineage = ArtifactLayoutResolver.infer_lineage(pred_file)
        assert lineage == ("hyperrag", "narrativeqa", "run-001")

    def test_flat_lineage_is_none(self, tmp_path):
        pred_file = tmp_path / "evaluation" / "predictions" / "predictions_hotpotqa.json"
        lineage = ArtifactLayoutResolver.infer_lineage(pred_file)
        assert lineage == (None, None, None)

    def test_non_standard_path_raises(self, tmp_path):
        bad = tmp_path / "results" / "output.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("[]")
        with pytest.raises(ArtifactPathError):
            ArtifactLayoutResolver.infer_run_root(bad)

    def test_too_short_path_raises(self, tmp_path):
        short = tmp_path / "data.json"
        short.write_text("[]")
        with pytest.raises(ArtifactPathError):
            ArtifactLayoutResolver.infer_run_root(short)


class TestThreeLayerLayout:
    def test_predictions_three_layer(self):
        root = Path("/tmp/eval_root")
        layout = ArtifactLayoutResolver.predictions(
            root, "narrativeqa", project="hyperrag", source_run_id="run-001"
        )
        assert (
            layout.predictions_dir
            == root / "evaluation" / "hyperrag" / "narrativeqa" / "run-001" / "predictions"
        )
        assert layout.predictions_file.name == "predictions_narrativeqa.json"

    def test_judge_run_three_layer(self):
        root = Path("/tmp/eval_root")
        layout = ArtifactLayoutResolver.judge_run(
            root,
            "qwen-model",
            "run-1",
            project="hyperrag",
            dataset="narrativeqa",
            source_run_id="run-001",
        )
        assert layout.judge_run_dir == (
            root
            / "evaluation"
            / "hyperrag"
            / "narrativeqa"
            / "run-001"
            / "llmjudge"
            / "qwen-model"
            / "run-1"
        )

    def test_partial_lineage_raises(self):
        root = Path("/tmp/eval_root")
        with pytest.raises(ArtifactPathError, match="all-or-nothing"):
            ArtifactLayoutResolver.predictions(root, "ds", project="hyperrag")

    def test_find_judge_run_scoped_to_lineage(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(
            root, "qwen-model", "run1", project="hyperrag", dataset="ds", source_run_id="run-001"
        )
        JudgeArtifactStore().write_run_manifest(
            layout, {"judge_run_id": "run1", "judge_model": "qwen-model"}
        )
        # Found within the scoped lineage
        found = ArtifactLayoutResolver.find_judge_run(
            root, "run1", project="hyperrag", dataset="ds", source_run_id="run-001"
        )
        assert found is not None
        # Not found in a different lineage
        assert (
            ArtifactLayoutResolver.find_judge_run(
                root, "run1", project="lightrag", dataset="ds", source_run_id="run-001"
            )
            is None
        )


class TestHashing:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        digest = sha256_file(f)
        assert len(digest) == 64
        assert digest == sha256_file(f)

    def test_sha256_json_deterministic(self):
        d1 = sha256_json({"b": 1, "a": 2})
        d2 = sha256_json({"a": 2, "b": 1})
        assert d1 == d2


class TestJudgeArtifactStore:
    def test_write_predictions(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.predictions(root, "ds")
        store = JudgeArtifactStore()
        store.write_predictions(layout, [{"id": 0, "question": "q"}])
        assert layout.predictions_file.exists()

    def test_write_result_and_summary(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        store.write_result(layout, "generation", {"scores": {"a": 1.0}})
        store.write_summary(layout, {"run_id": "run1"})
        assert layout.generation_file.exists()
        assert layout.summary_file.exists()

    def test_update_latest(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        store.update_latest(layout, "run1")
        assert layout.latest_pointer.exists()
        data = json.loads(layout.latest_pointer.read_text())
        assert data["latest_run_id"] == "run1"

    def test_update_latest_atomic(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        store.update_latest(layout, "run1")
        content = layout.latest_pointer.read_text()
        data = json.loads(content)
        assert data["schema_version"] == 1

    def test_load_for_resume_nonexistent(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        assert store.load_for_resume(layout) is None

    def test_load_for_resume_existing(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        store.write_run_manifest(layout, {"judge_run_id": "run1"})
        data = store.load_for_resume(layout)
        assert data["judge_run_id"] == "run1"

    def test_find_judge_run_across_model_directories(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "qwen-model", "run1")
        JudgeArtifactStore().write_run_manifest(
            layout,
            {"judge_run_id": "run1", "judge_model": "qwen-model"},
        )
        found = ArtifactLayoutResolver.find_judge_run(root, "run1")
        assert found is not None
        found_layout, manifest = found
        assert found_layout.judge_model_dir.name == "qwen-model"
        assert manifest["judge_run_id"] == "run1"

    def test_find_judge_run_rejects_ambiguous_model_directories(self, tmp_path):
        root = tmp_path / "eval_root"
        store = JudgeArtifactStore()
        for model in ("model-a", "model-b"):
            layout = ArtifactLayoutResolver.judge_run(root, model, "run1")
            store.write_run_manifest(layout, {"judge_run_id": "run1"})
        with pytest.raises(ArtifactPathError, match="ambiguous"):
            ArtifactLayoutResolver.find_judge_run(root, "run1")

    def test_unknown_result_kind_raises(self, tmp_path):
        root = tmp_path / "eval_root"
        layout = ArtifactLayoutResolver.judge_run(root, "model", "run1")
        store = JudgeArtifactStore()
        with pytest.raises(ValueError):
            store.write_result(layout, "unknown", {})

    def test_path_validation_in_predictions(self):
        root = Path("/tmp/safe_root")
        layout = ArtifactLayoutResolver.predictions(root, "test")
        eval_dir = root / "evaluation"
        assert str(layout.predictions_file).startswith(str(eval_dir))
