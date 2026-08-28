"""Tests for JudgeEvaluationService — resume, merge summary, hash validation."""

import json

import pytest

from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
)
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import (
    JudgeConfigurationError,
    JudgeResumeConflictError,
)
from pipeline.evaluation.judge.evaluation_service import (
    JudgeEvaluationService,
    _select_samples_for_metric_update,
)
from pipeline.evaluation.judge.models import (
    EvaluationKind,
    JudgeSample,
    SampleEvaluationStatus,
)
from pipeline.evaluation.judge.runner import JudgeEvaluationRunner


class _FakeEvaluator:
    kind = EvaluationKind.GENERATION
    supported_metrics = ("qa_em", "qa_f1")
    default_metrics = supported_metrics

    async def evaluate(self, sample, llm, metrics, context_top_k=5):
        from pipeline.evaluation.judge.models import JudgeDetailedResult

        return JudgeDetailedResult(
            id=sample.id,
            seq=sample.seq,
            question=sample.question,
            ground_truth=sample.ground_truth,
            generated_answer=sample.answer,
            contexts=sample.contexts,
            metrics=dict.fromkeys(metrics, 1.0),
            llm_intermediate={},
            status=SampleEvaluationStatus.SUCCESS,
        )


class _RetEval:
    kind = EvaluationKind.RETRIEVAL
    supported_metrics = ("context_relevancy", "evidence_recall")
    default_metrics = supported_metrics

    async def evaluate(self, sample, llm, metrics, context_top_k=5):
        from pipeline.evaluation.judge.models import JudgeDetailedResult

        return JudgeDetailedResult(
            id=sample.id,
            seq=sample.seq,
            question=sample.question,
            ground_truth=sample.ground_truth,
            generated_answer=sample.answer,
            contexts=sample.contexts,
            metrics=dict.fromkeys(metrics, 0.9),
            llm_intermediate={},
            status=SampleEvaluationStatus.SUCCESS,
        )


class _FakeLLM:
    class _Config:
        model = "fake-model"
        base_url = "http://localhost:8080"

    config = _Config()


class _DifferentLLM:
    class _Config:
        model = "new-model"
        base_url = "http://localhost:9090"

    config = _Config()


@pytest.fixture
def predictions_file(tmp_path):
    p = tmp_path / "predictions.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "question": "Q1",
                    "answer": "A1",
                    "ground_truth": "A1",
                    "contexts": ["C1"],
                    "question_type": "qa",
                },
                {
                    "id": 1,
                    "question": "Q2",
                    "answer": "A2",
                    "ground_truth": "A2",
                    "contexts": ["C2"],
                    "question_type": "qa",
                },
            ]
        )
    )
    return p


@pytest.fixture
def service_and_layout(tmp_path, predictions_file):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "sample.json").write_text(
        json.dumps(
            [
                {
                    "question": "Q1",
                    "answer": "A1",
                    "paragraphs": [{"text": "E1", "is_supporting": True}],
                },
                {
                    "question": "Q2",
                    "answer": "A2",
                    "paragraphs": [{"text": "E2", "is_supporting": True}],
                },
            ]
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    resolver = ArtifactLayoutResolver()
    store = JudgeArtifactStore()
    runner = JudgeEvaluationRunner(max_concurrent=1)

    service = JudgeEvaluationService(
        llm=_FakeLLM(),
        runner=runner,
        ground_truth_repository=GroundTruthRepository(dataset_dir),
        resolver=resolver,
        store=store,
    )
    return service, artifact_root, resolver, store


@pytest.mark.asyncio
async def test_generation_then_retrieval_same_run_merges(
    service_and_layout, predictions_file, monkeypatch
):
    """Verify generation+retrieval keeps both qa_em and context_relevancy in summary."""
    service, artifact_root, resolver, store = service_and_layout
    run_id = "test_merge_run"

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.RetrievalSampleEvaluator",
        lambda *args, **kwargs: _RetEval(),
    )

    # Run generation
    manifest1 = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em", "qa_f1"),
        dataset="sample",
    )
    assert "generation" in manifest1.evaluation_kinds

    # Run retrieval — same run_id should merge, not error
    manifest2 = await service.run_retrieval(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("context_relevancy",),
        dataset="sample",
        resume_run_id=run_id,
    )
    assert "generation" in manifest2.evaluation_kinds
    assert "retrieval" in manifest2.evaluation_kinds
    assert len(manifest2.result_files) >= 2

    # Verify summary retains both generation and retrieval scores
    layout = resolver.judge_run(artifact_root, "fake-model", run_id)
    assert layout.summary_file.exists()
    summary = json.loads(layout.summary_file.read_text())
    assert "qa_em" in summary["average_scores"]
    assert "context_relevancy" in summary["average_scores"]
    # kind_counts should track both kinds separately
    assert "kind_generation" in summary["kind_counts"]
    assert "kind_retrieval" in summary["kind_counts"]
    gen_counts = summary["kind_counts"]["kind_generation"]
    assert gen_counts["total_samples"] == 2
    ret_counts = summary["kind_counts"]["kind_retrieval"]
    assert ret_counts["total_samples"] == 2
    assert set(manifest2.evaluation_parameters) == {
        "generation",
        "retrieval",
    }
    assert set(manifest2.kind_counts) == {"generation", "retrieval"}
    assert manifest2.total_samples == 4

    forced = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em",),
        dataset="sample",
        force=True,
    )
    assert forced.total_samples == 4
    assert set(forced.evaluation_parameters) == {
        "generation",
        "retrieval",
    }


@pytest.mark.asyncio
async def test_num_samples_limits_original_prediction_order(
    service_and_layout, predictions_file, monkeypatch
):
    service, artifact_root, resolver, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="top_k_run",
        metrics=("qa_em",),
        dataset="sample",
        num_samples=1,
    )

    layout = resolver.judge_run(artifact_root, "fake-model", "top_k_run")
    result = json.loads(layout.generation_file.read_text())
    assert manifest.total_samples == 1
    assert result["total_samples"] == 1
    assert [item["question"] for item in result["detailed"]] == ["Q1"]


@pytest.mark.asyncio
async def test_resume_uses_persisted_model_directory_when_execution_model_changes(
    service_and_layout, predictions_file, monkeypatch
):
    service, artifact_root, resolver, _ = service_and_layout
    run_id = "cross_model_resume"
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em",),
        dataset="sample",
    )

    service._llm = _DifferentLLM()
    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id=run_id,
    )

    layout = resolver.judge_run(artifact_root, "fake-model", run_id)
    result = json.loads(layout.generation_file.read_text())
    assert manifest.judge_model == "fake-model"
    assert manifest.metric_updates[-1]["execution_model"] == "new-model"
    assert all(item["metrics"] == {"qa_em": 1.0, "qa_f1": 1.0} for item in result["detailed"])
    assert not resolver.judge_run(artifact_root, "new-model", run_id).run_manifest_file.exists()


@pytest.mark.asyncio
async def test_same_kind_without_force_raises(service_and_layout, predictions_file, monkeypatch):
    service, artifact_root, _, _ = service_and_layout
    run_id = "test_run"

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    # First run
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em",),
        dataset="sample",
    )

    # Second run with same kind, no force -> should raise
    with pytest.raises(JudgeResumeConflictError):
        await service.run_generation(
            predictions_file=predictions_file,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id=run_id,
            metrics=("qa_em",),
            dataset="sample",
        )


@pytest.mark.asyncio
async def test_resume_fills_missing_metric_without_overwriting_existing(
    service_and_layout, predictions_file, monkeypatch
):
    service, artifact_root, resolver, _ = service_and_layout
    run_id = "metric_fill"
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em",),
        dataset="sample",
    )
    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id=run_id,
    )

    layout = resolver.judge_run(artifact_root, "fake-model", run_id)
    result = json.loads(layout.generation_file.read_text())
    assert all(item["metrics"] == {"qa_em": 1.0, "qa_f1": 1.0} for item in result["detailed"])
    assert manifest.metric_updates[-1]["mode"] == "fill_missing"


@pytest.mark.asyncio
async def test_force_metrics_replaces_only_requested_metric(
    service_and_layout, predictions_file, monkeypatch
):
    service, artifact_root, resolver, _ = service_and_layout
    run_id = "metric_replace"
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em", "qa_f1"),
        dataset="sample",
    )

    class _ReplacementEvaluator(_FakeEvaluator):
        async def evaluate(self, sample, llm, metrics, context_top_k=5):
            result = await super().evaluate(sample, llm, metrics, context_top_k)
            result.metrics = dict.fromkeys(metrics, 0.25)
            return result

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _ReplacementEvaluator(),
    )
    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id=run_id,
        force_metrics=True,
    )

    layout = resolver.judge_run(artifact_root, "fake-model", run_id)
    result = json.loads(layout.generation_file.read_text())
    assert all(item["metrics"]["qa_em"] == 1.0 for item in result["detailed"])
    assert all(item["metrics"]["qa_f1"] == 0.25 for item in result["detailed"])
    assert manifest.metric_updates[-1]["mode"] == "force_metrics"


@pytest.mark.asyncio
async def test_resume_validates_hash(service_and_layout, predictions_file, monkeypatch):
    service, artifact_root, _, store = service_and_layout

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    # Create a first run
    run_id = "first_run_hash"
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=run_id,
        metrics=("qa_em",),
        dataset="sample",
    )

    # Try to resume with different predictions file -> hash mismatch
    other_file = predictions_file.parent / "other.json"
    other_file.write_text(json.dumps([{"id": 0, "question": "Q3"}]))
    with pytest.raises(JudgeResumeConflictError) as exc_info:
        await service.run_generation(
            predictions_file=other_file,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="new_run_hash",
            metrics=("qa_em",),
            dataset="sample",
            resume_run_id=run_id,
        )
    assert "SHA-256 mismatch" in str(exc_info.value)


@pytest.mark.asyncio
async def test_resume_validates_dataset_digest(service_and_layout, predictions_file, monkeypatch):
    service, artifact_root, _, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="dataset_digest_run",
        metrics=("qa_em",),
        dataset="sample",
    )
    dataset_path = service._ground_truth.resolve_dataset_path("sample")
    dataset_path.write_text(
        json.dumps(
            [
                {"question": "Q1", "answer": "changed", "paragraphs": []},
                {"question": "Q2", "answer": "A2", "paragraphs": []},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(JudgeResumeConflictError, match="Dataset digest mismatch"):
        await service.run_generation(
            predictions_file=predictions_file,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="unused",
            metrics=("qa_em",),
            dataset="sample",
            resume_run_id="dataset_digest_run",
        )


@pytest.mark.asyncio
async def test_resume_validates_metric_parameters(
    service_and_layout, predictions_file, monkeypatch
):
    service, artifact_root, _, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="metric_parameter_run",
        metrics=("qa_em",),
        context_top_k=5,
        dataset="sample",
    )
    with pytest.raises(JudgeResumeConflictError, match="Metric parameter mismatch"):
        await service.run_generation(
            predictions_file=predictions_file,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="unused",
            metrics=("qa_em",),
            context_top_k=3,
            dataset="sample",
            resume_run_id="metric_parameter_run",
        )


@pytest.mark.asyncio
async def test_resume_adding_metric_keeps_existing_metric_parameters(
    service_and_layout, predictions_file, monkeypatch
):
    """Adding a metric to an existing kind must not drop the earlier metric's
    recorded parameters, since its scores survive the result merge."""
    service, artifact_root, _, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="metric_parameter_merge_run",
        metrics=("qa_em",),
        dataset="sample",
    )
    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id="metric_parameter_merge_run",
    )
    assert set(manifest.metric_parameters["generation"]) == {"qa_em", "qa_f1"}


@pytest.mark.asyncio
async def test_force_replaces_metric_parameters(service_and_layout, predictions_file, monkeypatch):
    """``--force`` recomputes the kind from scratch, so parameters for metrics
    no longer present in the results must not linger in the manifest."""
    service, artifact_root, _, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="metric_parameter_force_run",
        metrics=("qa_em",),
        dataset="sample",
    )
    manifest = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id="metric_parameter_force_run",
        force=True,
    )
    assert set(manifest.metric_parameters["generation"]) == {"qa_f1"}


@pytest.mark.asyncio
async def test_resume_run_id_different_from_new_run_id(
    service_and_layout, predictions_file, monkeypatch
):
    """Item 1: When resume_run_id != run_id, the existing run is resumed, not
    the new timestamp directory."""
    service, artifact_root, resolver, store = service_and_layout

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    # Create an initial run
    initial_run_id = "initial_run_20240101"
    manifest1 = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=initial_run_id,
        metrics=("qa_em",),
        dataset="sample",
    )
    assert manifest1.evaluation_kinds == ["generation"]

    # Now resume that run with a DIFFERENT new run_id
    new_run_id = "new_timestamp_20240102"
    resumed_generation = await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=new_run_id,
        metrics=("qa_em",),
        dataset="sample",
        resume_run_id=initial_run_id,
    )
    assert resumed_generation.judge_run_id == initial_run_id
    assert resumed_generation.evaluation_kinds == ["generation"]

    async def _finite_context_relevance(*args, **kwargs):
        return 0.5

    monkeypatch.setattr(
        "pipeline.evaluation.judge.metrics.context_relevance.compute_context_relevance",
        _finite_context_relevance,
    )

    manifest2 = await service.run_retrieval(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id=new_run_id,
        metrics=("context_relevancy",),
        dataset="sample",
        resume_run_id=initial_run_id,
    )
    # Must use the original run_id (initial), not the new one
    assert manifest2.judge_run_id == initial_run_id
    assert "generation" in manifest2.evaluation_kinds
    assert "retrieval" in manifest2.evaluation_kinds

    # The new run_id directory should NOT exist
    new_layout = resolver.judge_run(artifact_root, "fake-model", new_run_id)
    assert not new_layout.judge_run_dir.exists() or not new_layout.run_manifest_file.exists()

    # The existing run should have both results
    layout = resolver.judge_run(artifact_root, "fake-model", initial_run_id)
    assert layout.run_manifest_file.exists()
    assert layout.generation_file.exists()
    assert layout.retrieval_file.exists()


@pytest.mark.asyncio
async def test_resume_nonexistent_run_raises_manifest_not_found(
    service_and_layout, predictions_file
):
    """Item 1: Resuming a nonexistent run fails with 'manifest not found'."""
    service, artifact_root, _, _ = service_and_layout

    with pytest.raises(JudgeResumeConflictError) as exc_info:
        await service.run_generation(
            predictions_file=predictions_file,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="new_run",
            metrics=("qa_em",),
            dataset="sample",
            resume_run_id="nonexistent_run",
        )
    assert "manifest not found" in str(exc_info.value).lower()


def test_indexing_new_resume_conflict_and_force(service_and_layout, tmp_path, monkeypatch):
    service, artifact_root, resolver, _ = service_and_layout
    base_path = tmp_path / "graph"
    base_path.mkdir()
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {"entity_count": 3.0},
    )

    result = service.run_indexing(
        framework="graphrag",
        base_path=base_path,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="index_run",
    )
    assert result["metrics"]["entity_count"] == 3.0
    layout = resolver.judge_run(artifact_root, "fake-model", "index_run")
    assert layout.indexing_file.exists()
    assert layout.run_manifest_file.exists()
    assert layout.summary_file.exists()
    assert layout.latest_pointer.exists()

    with pytest.raises(JudgeResumeConflictError):
        service.run_indexing(
            framework="graphrag",
            base_path=base_path,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="index_run",
        )

    resumed = service.run_indexing(
        framework="graphrag",
        base_path=base_path,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="new_timestamp",
        resume_run_id="index_run",
    )
    assert resumed["metrics"]["entity_count"] == 3.0

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {"entity_count": 4.0},
    )
    forced = service.run_indexing(
        framework="graphrag",
        base_path=base_path,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="index_run",
        force=True,
    )
    assert forced["metrics"]["entity_count"] == 4.0


def test_indexing_rejects_empty_invalid_and_changed_resume_input(
    service_and_layout, tmp_path, monkeypatch
):
    service, artifact_root, _, _ = service_and_layout
    base_path = tmp_path / "graph_a"
    base_path.mkdir()
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {},
    )
    with pytest.raises(JudgeConfigurationError, match="No graph data"):
        service.run_indexing(
            framework="graphrag",
            base_path=base_path,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="empty_index",
        )

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {"entity_count": float("inf")},
    )
    with pytest.raises(JudgeConfigurationError, match="Invalid indexing metrics"):
        service.run_indexing(
            framework="graphrag",
            base_path=base_path,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="invalid_index",
        )

    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {"entity_count": 1.0},
    )
    service.run_indexing(
        framework="graphrag",
        base_path=base_path,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="input_contract_run",
    )
    changed_path = tmp_path / "graph_b"
    changed_path.mkdir()
    with pytest.raises(JudgeResumeConflictError, match="Indexing input mismatch"):
        service.run_indexing(
            framework="graphrag",
            base_path=changed_path,
            artifact_run_root=artifact_root,
            judge_model="fake-model",
            run_id="unused",
            resume_run_id="input_contract_run",
        )


@pytest.mark.asyncio
async def test_indexing_appends_to_generation_run(
    service_and_layout, predictions_file, tmp_path, monkeypatch
):
    service, artifact_root, resolver, _ = service_and_layout
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )
    await service.run_generation(
        predictions_file=predictions_file,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="combined_run",
        metrics=("qa_em",),
        dataset="sample",
    )
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.calculate_indexing_metrics",
        lambda **kwargs: {"entity_count": 2.0},
    )
    service.run_indexing(
        framework="graphrag",
        base_path=tmp_path,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused_new_id",
        resume_run_id="combined_run",
    )
    layout = resolver.judge_run(artifact_root, "fake-model", "combined_run")
    manifest = json.loads(layout.run_manifest_file.read_text())
    assert manifest["evaluation_kinds"] == ["generation", "indexing"]
    assert set(manifest["evaluation_parameters"]) == {"generation", "indexing"}


@pytest.mark.asyncio
async def test_duplicate_questions_with_distinct_canonical_ids_survive_resume(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "sample.json").write_text(
        json.dumps(
            [
                {
                    "id": "source-0",
                    "question": "Qdup",
                    "answer": "GT0",
                    "paragraphs": [{"text": "E0", "is_supporting": True}],
                },
                {
                    "id": "source-1",
                    "question": "Qdup",
                    "answer": "GT1",
                    "paragraphs": [{"text": "E1", "is_supporting": True}],
                },
            ]
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    p = tmp_path / "preds_dup.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "question": "Qdup",
                    "answer": "A1",
                    "ground_truth": "",
                    "contexts": ["C1"],
                    "question_type": "qa",
                },
                {
                    "id": 1,
                    "question": "Qdup",
                    "answer": "A2",
                    "ground_truth": "",
                    "contexts": ["C2"],
                    "question_type": "qa",
                },
            ]
        ),
        encoding="utf-8",
    )
    resolver = ArtifactLayoutResolver()
    store = JudgeArtifactStore()
    service = JudgeEvaluationService(
        llm=_FakeLLM(),
        runner=JudgeEvaluationRunner(max_concurrent=1),
        ground_truth_repository=GroundTruthRepository(dataset_dir),
        resolver=resolver,
        store=store,
    )
    monkeypatch.setattr(
        "pipeline.evaluation.judge.evaluation_service.GenerationSampleEvaluator",
        lambda: _FakeEvaluator(),
    )

    first = await service.run_generation(
        predictions_file=p,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="dup_run",
        metrics=("qa_em",),
        dataset="sample",
    )
    assert first.total_samples == 2

    resumed = await service.run_generation(
        predictions_file=p,
        artifact_run_root=artifact_root,
        judge_model="fake-model",
        run_id="unused",
        metrics=("qa_f1",),
        dataset="sample",
        resume_run_id="dup_run",
    )
    assert resumed.total_samples == 2

    layout = resolver.judge_run(artifact_root, "fake-model", "dup_run")
    result = json.loads(layout.generation_file.read_text())
    assert result["total_samples"] == 2
    assert {d["seq"] for d in result["detailed"]} == {0, 1}
    assert {d["ground_truth"] for d in result["detailed"]} == {"GT0", "GT1"}
    assert {d["generated_answer"] for d in result["detailed"]} == {"A1", "A2"}
    assert all(d["metrics"] == {"qa_em": 1.0, "qa_f1": 1.0} for d in result["detailed"])


def test_retry_failed_selects_only_previously_failed_samples():
    sample = JudgeSample(
        id=7,
        seq=7,
        question="Q",
        answer="A",
        ground_truth="A",
        contexts=["C"],
    )
    existing = {
        "detailed": [
            {
                "id": 7,
                "seq": 7,
                "status": SampleEvaluationStatus.SAMPLE_FAILED.value,
                "metrics": {},
            }
        ]
    }

    assert (
        _select_samples_for_metric_update(
            [sample],
            EvaluationKind.RETRIEVAL,
            ("context_relevancy",),
            existing,
            force_metrics=False,
            retry_failed=False,
        )
        == []
    )
    assert _select_samples_for_metric_update(
        [sample],
        EvaluationKind.RETRIEVAL,
        ("context_relevancy",),
        existing,
        force_metrics=False,
        retry_failed=True,
    ) == [sample]
