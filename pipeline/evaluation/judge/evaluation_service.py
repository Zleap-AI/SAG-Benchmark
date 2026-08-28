"""Judge evaluation application services.

JudgeEvaluationService orchestrates reading predictions, selecting evaluators,
calling the runner, and writing results/manifests/summaries.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.core.ai.base import BaseLLMClient
from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
    sanitize_path_component,
    sha256_file,
)
from pipeline.evaluation.judge.dataset_adapters.models import DatasetCapability
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import (
    ArtifactPathError,
    DatasetError,
    JudgeConfigurationError,
    JudgeResumeConflictError,
    MetricResultError,
    PredictionValidationError,
)
from pipeline.evaluation.judge.generation import (
    GenerationSampleEvaluator,
    metrics_for_question_type,
)
from pipeline.evaluation.judge.indexing import calculate_indexing_metrics
from pipeline.evaluation.judge.metric_validation import validate_metric_mapping
from pipeline.evaluation.judge.models import (
    EvaluationKind,
    FileDigest,
    JudgeRunManifest,
    JudgeRunParameters,
    JudgeRunStatus,
    JudgeRunSummary,
    JudgeSample,
    MetricRunParameters,
    SampleEvaluationStatus,
)
from pipeline.evaluation.judge.retrieval import RetrievalSampleEvaluator
from pipeline.evaluation.judge.runner import JudgeEvaluationRunner
from pipeline.utils import get_logger

logger = get_logger(__name__)


class JudgeEvaluationService:
    """Application service for Judge evaluation runs."""

    def __init__(
        self,
        llm: BaseLLMClient | None,
        runner: JudgeEvaluationRunner,
        ground_truth_repository: GroundTruthRepository,
        resolver: ArtifactLayoutResolver,
        store: JudgeArtifactStore,
    ) -> None:
        self._llm = llm
        self._runner = runner
        self._ground_truth = ground_truth_repository
        self._resolver = resolver
        self._store = store

    async def run_generation(
        self,
        predictions_file: Path,
        artifact_run_root: Path,
        judge_model: str,
        run_id: str,
        metrics: tuple[str, ...] = (),
        context_top_k: int = 5,
        num_samples: int | None = None,
        force: bool = False,
        force_metrics: bool = False,
        retry_failed: bool = False,
        dataset: str | None = None,
        resume_run_id: str | None = None,
    ) -> JudgeRunManifest:
        """Run generation evaluation and persist results."""
        return await self._run_kind(
            kind=EvaluationKind.GENERATION,
            predictions_file=predictions_file,
            artifact_run_root=artifact_run_root,
            judge_model=judge_model,
            run_id=run_id,
            metrics=metrics,
            context_top_k=context_top_k,
            num_samples=num_samples,
            force=force,
            force_metrics=force_metrics,
            retry_failed=retry_failed,
            dataset=dataset,
            resume_run_id=resume_run_id,
        )

    async def run_retrieval(
        self,
        predictions_file: Path,
        artifact_run_root: Path,
        judge_model: str,
        run_id: str,
        metrics: tuple[str, ...] = (),
        context_top_k: int = 5,
        num_samples: int | None = None,
        force: bool = False,
        force_metrics: bool = False,
        retry_failed: bool = False,
        dataset: str | None = None,
        resume_run_id: str | None = None,
    ) -> JudgeRunManifest:
        """Run retrieval evaluation and persist results."""
        return await self._run_kind(
            kind=EvaluationKind.RETRIEVAL,
            predictions_file=predictions_file,
            artifact_run_root=artifact_run_root,
            judge_model=judge_model,
            run_id=run_id,
            metrics=metrics,
            context_top_k=context_top_k,
            num_samples=num_samples,
            force=force,
            force_metrics=force_metrics,
            retry_failed=retry_failed,
            dataset=dataset,
            resume_run_id=resume_run_id,
        )

    def run_indexing(
        self,
        framework: str,
        base_path: Path,
        artifact_run_root: Path,
        judge_model: str,
        run_id: str,
        folder_name: str | None = None,
        resume_run_id: str | None = None,
        force: bool = False,
        project: str | None = None,
        dataset: str | None = None,
        source_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Run indexing evaluation and persist results."""
        indexing_input = _indexing_input_contract(
            framework=framework,
            base_path=base_path,
            folder_name=folder_name,
            project=project,
            dataset=dataset,
            source_run_id=source_run_id,
        )
        # Resolve layout for resume run first if applicable
        if resume_run_id:
            resume_layout = self._resolver.judge_run(
                artifact_run_root,
                judge_model,
                resume_run_id,
                project=project,
                dataset=dataset,
                source_run_id=source_run_id,
            )
            existing = self._store.load_for_resume(resume_layout)
            if not existing:
                raise JudgeResumeConflictError(
                    f"Cannot resume indexing run {resume_run_id}: run manifest not found"
                )
            if "indexing" in existing.get("evaluation_kinds", []):
                _validate_indexing_resume(
                    existing,
                    indexing_input,
                    allow_input_change=force,
                )
            layout = resume_layout
            effective_run_id = resume_run_id
        else:
            layout = self._resolver.judge_run(
                artifact_run_root,
                judge_model,
                run_id,
                project=project,
                dataset=dataset,
                source_run_id=source_run_id,
            )
            existing = self._store.load_for_resume(layout)
            effective_run_id = run_id

        # Check same-kind conflict
        if existing and "indexing" in existing.get("evaluation_kinds", []) and not force:
            if resume_run_id:
                if not layout.indexing_file.exists():
                    raise JudgeResumeConflictError(
                        f"Indexing manifest exists but result is missing: {layout.indexing_file}"
                    )
                with open(layout.indexing_file, encoding="utf-8") as f:
                    return json.load(f)
            raise JudgeResumeConflictError(
                f"Indexing already exists in run {effective_run_id}; use --force to overwrite"
            )

        metrics = calculate_indexing_metrics(
            framework=framework,
            base_path=str(base_path),
            folder_name=folder_name,
        )
        if not metrics:
            raise JudgeConfigurationError(
                f"No graph data found for {framework} in {base_path}; no indexing result was written"
            )
        try:
            validate_metric_mapping(metrics, scope="indexing metrics")
        except MetricResultError as exc:
            raise JudgeConfigurationError(f"Invalid indexing metrics: {exc}") from exc
        result_data = {
            "framework": framework,
            "base_path": str(base_path),
            "folder_name": folder_name,
            "metrics": metrics,
            "input_contract": indexing_input,
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._store.write_result(layout, "indexing", result_data)

        completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        indexing_parameters = JudgeRunParameters(
            metrics=sorted(metrics),
            force=force,
        )
        if existing:
            kinds = list(existing.get("evaluation_kinds", []))
            if "indexing" not in kinds:
                kinds.append("indexing")
            result_files = dict(existing.get("result_files", {}))
            result_files["indexing"] = "indexing_results.json"
            evaluation_parameters = dict(existing.get("evaluation_parameters", {}))
            evaluation_parameters["indexing"] = indexing_parameters.model_dump()
            kind_counts = dict(existing.get("kind_counts", {}))
            kind_counts["indexing"] = {
                "total_samples": 0,
                "successful_samples": 0,
                "failed_samples": 0,
            }
            manifest_data = {
                **existing,
                "evaluation_kinds": kinds,
                "parameters": indexing_parameters.model_dump(),
                "evaluation_parameters": evaluation_parameters,
                "kind_counts": kind_counts,
                "result_files": result_files,
                "indexing_input": indexing_input,
                "completed_at": completed_at,
            }
        else:
            manifest_data = JudgeRunManifest(
                schema_version=2,
                judge_run_id=effective_run_id,
                status=JudgeRunStatus.SUCCESS,
                started_at=completed_at,
                completed_at=completed_at,
                judge_model=judge_model,
                base_url_display="none",
                evaluation_kinds=["indexing"],
                parameters=indexing_parameters,
                evaluation_parameters={"indexing": indexing_parameters},
                kind_counts={
                    "indexing": {
                        "total_samples": 0,
                        "successful_samples": 0,
                        "failed_samples": 0,
                    }
                },
                git_commit=_git_commit(),
                git_dirty=_git_dirty(),
                python_version=sys.version.split()[0],
                result_files={"indexing": "indexing_results.json"},
                indexing_input=indexing_input,
            ).model_dump()
        self._store.write_run_manifest(layout, manifest_data)

        summary_data: dict[str, Any] = {}
        if layout.summary_file.exists():
            with open(layout.summary_file, encoding="utf-8") as f:
                summary_data = json.load(f)
        summary_data["judge_run_id"] = effective_run_id
        summary_data["indexing_metrics"] = metrics
        summary_kind_counts = dict(summary_data.get("kind_counts", {}))
        summary_kind_counts["kind_indexing"] = {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
        }
        summary_data["kind_counts"] = summary_kind_counts
        self._store.write_summary(layout, summary_data)
        self._store.update_latest(layout, effective_run_id)
        return result_data

    async def _run_kind(
        self,
        kind: EvaluationKind,
        predictions_file: Path,
        artifact_run_root: Path,
        judge_model: str,
        run_id: str,
        metrics: tuple[str, ...],
        context_top_k: int,
        num_samples: int | None,
        force: bool,
        force_metrics: bool,
        retry_failed: bool,
        dataset: str | None,
        resume_run_id: str | None = None,
    ) -> JudgeRunManifest:
        # Resolve layout for the *resume* run first, not the new run_id
        project, dataset_id, source_run_id = self._resolver.infer_lineage(predictions_file)
        effective_dataset = dataset or dataset_id
        if resume_run_id:
            resume_layout = self._resolver.judge_run(
                artifact_run_root,
                judge_model,
                resume_run_id,
                project=project,
                dataset=dataset_id,
                source_run_id=source_run_id,
            )
            existing = self._store.load_for_resume(resume_layout)
            predictions_sha = sha256_file(predictions_file)
            effective_run_id = resume_run_id
            layout = resume_layout
        else:
            layout = self._resolver.judge_run(
                artifact_run_root,
                judge_model,
                run_id,
                project=project,
                dataset=dataset_id,
                source_run_id=source_run_id,
            )
            predictions_sha = sha256_file(predictions_file)
            existing = self._store.load_for_resume(layout)
            effective_run_id = run_id

        if force and force_metrics:
            raise JudgeConfigurationError("--force and --force-metrics are mutually exclusive")
        if force_metrics and not resume_run_id:
            raise JudgeConfigurationError("--force-metrics requires --resume-run-id")

        kind_exists = bool(existing and kind.value in existing.get("evaluation_kinds", []))
        if existing and not resume_run_id and not force:
            raise JudgeResumeConflictError(
                f"Judge run {effective_run_id} already exists; use a new --judge-run-id "
                "or --resume-run-id to update it"
            )

        existing_kind_result: dict[str, Any] | None = None
        if kind_exists and resume_run_id and not force:
            result_path = (
                layout.generation_file
                if kind == EvaluationKind.GENERATION
                else layout.retrieval_file
            )
            if not result_path.exists():
                raise JudgeResumeConflictError(
                    f"{kind.value} manifest exists but result is missing: {result_path}"
                )
            try:
                with open(result_path, encoding="utf-8") as f:
                    existing_kind_result = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise JudgeResumeConflictError(
                    f"Cannot load existing {kind.value} result: {result_path}: {exc}"
                ) from exc

        # Load predictions
        with open(predictions_file, encoding="utf-8") as f:
            raw = json.load(f)
        samples = [JudgeSample.from_predictions_row(r) for r in raw]
        # seq = predictions 文件内的行号,是样本的唯一身份;不同于 id(源数据
        # 集索引,不同源 id 可能拼出同一条问句而撞车,如 MusiQue)。截断必须
        # 在赋 seq 之后,保证 resume 时同一条样本仍匹配到同一 seq。
        for i, sample in enumerate(samples):
            sample.seq = i
        if num_samples:
            samples = samples[:num_samples]

        # Select evaluator
        requested_metrics_explicit = bool(metrics)
        if kind == EvaluationKind.GENERATION:
            evaluator = GenerationSampleEvaluator()
        else:
            evaluator = RetrievalSampleEvaluator(
                evidence_repository=self._ground_truth,
                dataset=effective_dataset or "",
            )

        if force_metrics and not requested_metrics_explicit:
            raise JudgeConfigurationError("--force-metrics requires explicit metrics")

        if not metrics:
            metrics = evaluator.default_metrics
        else:
            illegal = [m for m in metrics if m not in evaluator.supported_metrics]
            if illegal:
                raise JudgeConfigurationError(
                    f"Invalid {kind.value} metrics: {illegal}. "
                    f"Supported: {sorted(evaluator.supported_metrics)}"
                )

        dataset_descriptor = (
            self._ground_truth.descriptor(effective_dataset) if effective_dataset else None
        )
        dataset_files = _collect_dataset_digests(self._ground_truth, effective_dataset)
        conversion_adapter, conversion_adapter_version = _conversion_adapter_contract(
            predictions_file
        )
        requested_metric_parameters = {
            metric: MetricRunParameters(
                context_top_k=context_top_k,
                num_samples=num_samples,
            )
            for metric in metrics
        }
        if resume_run_id:
            _validate_resume(
                existing,
                resume_run_id,
                run_id,
                judge_model,
                predictions_sha,
                dataset_descriptor=dataset_descriptor,
                dataset_files=dataset_files,
                conversion_adapter=conversion_adapter,
                conversion_adapter_version=conversion_adapter_version,
                kind=kind,
                metric_parameters=requested_metric_parameters,
                allow_parameter_change=force or force_metrics,
            )

        if effective_dataset:
            samples = _canonicalize_samples(samples, self._ground_truth, effective_dataset)
        elif kind == EvaluationKind.RETRIEVAL and "evidence_recall" in metrics:
            raise JudgeConfigurationError(
                "--dataset is required when retrieval metric evidence_recall is enabled"
            )

        if kind == EvaluationKind.RETRIEVAL and "evidence_recall" in metrics:
            if effective_dataset is None:
                raise JudgeConfigurationError(
                    "--dataset is required when retrieval metric evidence_recall is enabled"
                )
            self._ground_truth.require_capability(
                effective_dataset, DatasetCapability.EVIDENCE_RECALL
            )

        if kind == EvaluationKind.GENERATION:
            samples = [
                sample
                for sample in samples
                if metrics_for_question_type(sample.question_type, metrics)
            ]

        if existing_kind_result is not None:
            samples = _select_samples_for_metric_update(
                samples=samples,
                kind=kind,
                metrics=metrics,
                existing_result=existing_kind_result,
                force_metrics=force_metrics,
                retry_failed=retry_failed,
            )
            if not samples:
                logger.info(
                    "Run %s already contains valid %s metrics %s; nothing to update",
                    effective_run_id,
                    kind.value,
                    list(metrics),
                )
                return JudgeRunManifest.model_validate(existing)

        evaluated_sample_count = len(samples)

        # Determine checkpoint path
        ckpt_path = (
            layout.generation_partial
            if kind == EvaluationKind.GENERATION
            else layout.retrieval_partial
        )

        # Run
        summary = await self._runner.run(
            samples=samples,
            evaluator=evaluator,
            llm=self._llm,
            metrics=metrics,
            context_top_k=context_top_k,
            checkpoint_path=str(ckpt_path),
            retry_failed=retry_failed,
        )

        # Persist result
        if existing_kind_result is not None:
            result_data = _merge_kind_results(
                existing=existing_kind_result,
                new_summary=summary,
                metrics=metrics,
                force_metrics=force_metrics,
            )
            summary = JudgeRunSummary.model_validate(result_data)
        else:
            result_data = _summary_to_dict(summary)
        _validate_persisted_metric_results(result_data, kind.value)
        merged_summary = _merge_summary(layout, summary, kind, effective_run_id)
        validate_metric_mapping(
            merged_summary.get("average_scores", {}),
            scope="persisted merged summary",
        )
        self._store.write_result(layout, kind.value, result_data)

        # Clean partial
        if ckpt_path.exists():
            ckpt_path.unlink()

        # Build or merge manifest
        new_started_at = summary.start_time or (existing.get("started_at", "") if existing else "")
        new_kinds = list(existing.get("evaluation_kinds", [])) if existing else []
        if kind.value not in new_kinds:
            new_kinds.append(kind.value)
        new_result_files = dict(existing.get("result_files", {})) if existing else {}
        new_result_files[kind.value] = f"{kind.value}_results.json"

        parameters = JudgeRunParameters(
            metrics=list(metrics),
            max_concurrent=self._runner._max_concurrent,
            context_top_k=context_top_k,
            num_samples=num_samples,
            force=force,
            force_metrics=force_metrics,
            retry_failed=retry_failed,
        )
        evaluation_parameters = dict(existing.get("evaluation_parameters", {}) if existing else {})
        evaluation_parameters[kind.value] = parameters
        kind_counts = dict(existing.get("kind_counts", {}) if existing else {})
        kind_counts[kind.value] = {
            "total_samples": summary.total_samples,
            "successful_samples": summary.successful_samples,
            "failed_samples": summary.failed_samples,
        }
        total_samples = sum(v["total_samples"] for v in kind_counts.values())
        successful_samples = sum(v["successful_samples"] for v in kind_counts.values())
        failed_samples = sum(v["failed_samples"] for v in kind_counts.values())
        if kind == EvaluationKind.RETRIEVAL:
            evidence_dataset_files = dataset_files
        else:
            evidence_dataset_files = list(
                existing.get("evidence_dataset_files", []) if existing else []
            )

        metric_parameters = dict(existing.get("metric_parameters", {}) if existing else {})
        if existing_kind_result is not None:
            # Results were merged into the existing kind, so metrics computed by
            # earlier runs survive in the artifact; keep their parameters too.
            kind_metric_parameters = dict(metric_parameters.get(kind.value, {}))
            kind_metric_parameters.update(requested_metric_parameters)
            metric_parameters[kind.value] = kind_metric_parameters
        else:
            metric_parameters[kind.value] = requested_metric_parameters
        manifest = JudgeRunManifest(
            schema_version=2,
            judge_run_id=effective_run_id,
            status=JudgeRunStatus.SUCCESS if failed_samples == 0 else JudgeRunStatus.PARTIAL,
            started_at=new_started_at,
            completed_at=summary.end_time or "",
            judge_model=(existing.get("judge_model", judge_model) if existing else judge_model),
            base_url_display=_sanitize_base_url(self._llm),
            predictions_path=str(predictions_file),
            predictions_sha256=predictions_sha,
            dataset=effective_dataset or "",
            dataset_adapter=(dataset_descriptor.name if dataset_descriptor else ""),
            dataset_adapter_version=(
                dataset_descriptor.adapter_version if dataset_descriptor else ""
            ),
            dataset_files=dataset_files,
            conversion_adapter=conversion_adapter,
            conversion_adapter_version=conversion_adapter_version,
            evaluation_kinds=new_kinds,
            evidence_dataset_files=evidence_dataset_files,
            parameters=parameters,
            evaluation_parameters=evaluation_parameters,
            metric_parameters=metric_parameters,
            kind_counts=kind_counts,
            git_commit=_git_commit(),
            git_dirty=_git_dirty(),
            python_version=sys.version.split()[0],
            total_samples=total_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            result_files=new_result_files,
            metric_updates=[
                *(existing.get("metric_updates", []) if existing else []),
                {
                    "kind": kind.value,
                    "metrics": list(metrics),
                    "mode": (
                        "force_kind"
                        if force
                        else "force_metrics"
                        if force_metrics
                        else "fill_missing"
                        if existing_kind_result is not None
                        else "new_kind"
                    ),
                    "evaluated_samples": evaluated_sample_count,
                    "updated_at": summary.end_time or "",
                    "execution_model": _execution_model(self._llm),
                },
            ],
        )
        self._store.write_run_manifest(layout, manifest.model_dump())

        # Merge summary (validated before any result/manifest writes above).
        self._store.write_summary(layout, merged_summary)

        # Update latest pointer
        self._store.update_latest(layout, effective_run_id)

        return manifest


def _canonicalize_samples(
    samples: list[JudgeSample],
    repository: GroundTruthRepository,
    dataset: str,
) -> list[JudgeSample]:
    """Attach canonical answers and evidence using the prediction row id."""
    canonical: list[JudgeSample] = []
    for sample in samples:
        try:
            entry = repository.match_canonical_id(dataset, sample.id, sample.question)
        except DatasetError as exc:
            raise PredictionValidationError(
                f"Dataset matching failed for row id {sample.id} in {dataset!r}, "
                f"question {sample.question!r}: {exc}"
            ) from exc
        sample.id = entry.id
        sample.ground_truth = entry.answer
        sample.evidences = list(entry.evidences)
        if not sample.source:
            sample.source = dataset
        canonical.append(sample)
    return canonical


def _detailed_uses_seq(existing_result: dict[str, Any]) -> bool:
    """Return whether persisted detailed entries carry row-seq identity.

    旧格式结果只有 `id`(源数据集索引,同问句会撞车),新格式还带 `seq`(predictions
    行号,保真每条样本)。按磁盘上整份文件的格式统一索引;格式切换只发生在整份
    文件层面,不会同文件混用。
    """
    for item in existing_result.get("detailed", []):
        if isinstance(item, dict) and item.get("seq") is not None:
            return True
    return False


def _metric_needs_run(value: Any) -> bool:
    """Return whether a persisted metric is missing or unusable."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return isinstance(value, float) and math.isnan(value)


def _sample_metrics(
    sample: JudgeSample,
    kind: EvaluationKind,
    metrics: tuple[str, ...],
) -> tuple[str, ...]:
    if kind == EvaluationKind.GENERATION:
        return metrics_for_question_type(sample.question_type, metrics)
    return metrics


def _select_samples_for_metric_update(
    samples: list[JudgeSample],
    kind: EvaluationKind,
    metrics: tuple[str, ...],
    existing_result: dict[str, Any],
    force_metrics: bool,
    retry_failed: bool,
) -> list[JudgeSample]:
    """Select only samples whose requested metrics need computation."""
    use_seq = _detailed_uses_seq(existing_result)
    old_index: dict[int, Any] = {}
    for item in existing_result.get("detailed", []):
        if not isinstance(item, dict):
            continue
        key = item.get("seq") if use_seq else item.get("id")
        if key is not None:
            old_index[key] = item
    selected: list[JudgeSample] = []
    for sample in samples:
        requested = _sample_metrics(sample, kind, metrics)
        if not requested:
            continue
        if force_metrics:
            selected.append(sample)
            continue
        key = sample.seq if use_seq else sample.id
        old = old_index.get(key)
        old_metrics = old.get("metrics", {}) if isinstance(old, dict) else {}
        if (
            isinstance(old, dict)
            and old.get("status") == SampleEvaluationStatus.SAMPLE_FAILED.value
            and not retry_failed
        ):
            continue
        if old is None or any(
            metric not in old_metrics or _metric_needs_run(old_metrics.get(metric))
            for metric in requested
        ):
            selected.append(sample)
    return selected


def _merge_kind_results(
    existing: dict[str, Any],
    new_summary: JudgeRunSummary,
    metrics: tuple[str, ...],
    force_metrics: bool,
) -> dict[str, Any]:
    """Merge metric-level updates without changing unrelated persisted scores."""
    use_seq = _detailed_uses_seq(existing)
    old_by_key: dict[int, dict[str, Any]] = {}
    for item in existing.get("detailed", []):
        if not isinstance(item, dict):
            continue
        key = item.get("seq") if use_seq else item.get("id")
        if key is not None:
            old_by_key[key] = dict(item)

    for new_result in new_summary.detailed:
        new = new_result.model_dump()
        key = new_result.seq if use_seq else new_result.id
        old = old_by_key.get(key)
        if old is None:
            old_by_key[key] = new
            continue

        old_scores = dict(old.get("metrics", {}))
        old_intermediate = dict(old.get("llm_intermediate") or {})
        new_intermediate = dict(new.get("llm_intermediate") or {})
        changed = False
        for metric in metrics:
            if metric not in new.get("metrics", {}):
                continue
            if force_metrics or metric not in old_scores or _metric_needs_run(old_scores[metric]):
                old_scores[metric] = new["metrics"][metric]
                changed = True
                if metric in new_intermediate:
                    old_intermediate[metric] = new_intermediate[metric]
                else:
                    old_intermediate.pop(metric, None)

        if changed:
            old["metrics"] = old_scores
            old["llm_intermediate"] = old_intermediate
            if all(not _metric_needs_run(value) for value in old_scores.values()):
                old["status"] = "success"
                old["error_type"] = None
                old["error_message"] = None
            else:
                old["status"] = new.get("status", "sample_failed")
                old["error_type"] = new.get("error_type")
                old["error_message"] = new.get("error_message")
        old_by_key[key] = old

    detailed = [old_by_key[key] for key in sorted(old_by_key)]
    all_metric_names = sorted({metric for item in detailed for metric in item.get("metrics", {})})
    average_scores: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    for metric in all_metric_names:
        values = [
            item["metrics"][metric]
            for item in detailed
            if metric in item.get("metrics", {}) and not _metric_needs_run(item["metrics"][metric])
        ]
        valid_counts[metric] = len(values)
        average_scores[metric] = sum(values) / len(values) if values else float("nan")

    successful = sum(1 for item in detailed if item.get("status") == "success")
    return {
        "average_scores": average_scores,
        "detailed": detailed,
        "total_tokens": existing.get("total_tokens", {}),
        "start_time": existing.get("start_time") or new_summary.start_time,
        "end_time": new_summary.end_time,
        "total_samples": len(detailed),
        "successful_samples": successful,
        "failed_samples": len(detailed) - successful,
        "metric_valid_counts": valid_counts,
    }


def _validate_resume(
    existing: dict[str, Any] | None,
    resume_run_id: str,
    new_run_id: str,
    judge_model: str,
    predictions_sha: str,
    *,
    dataset_descriptor: Any,
    dataset_files: list[FileDigest],
    conversion_adapter: str,
    conversion_adapter_version: str,
    kind: EvaluationKind,
    metric_parameters: dict[str, MetricRunParameters],
    allow_parameter_change: bool,
) -> None:
    """Validate that resume parameters match the existing run."""
    if not existing:
        raise JudgeResumeConflictError(f"Cannot resume run {resume_run_id}: run manifest not found")
    existing_model = existing.get("judge_model", "")
    if judge_model and existing_model:
        try:
            requested_path_model = sanitize_path_component(judge_model)
            existing_path_model = sanitize_path_component(existing_model)
        except ArtifactPathError as exc:
            raise JudgeResumeConflictError(f"Invalid Judge model path component: {exc}") from exc
        if requested_path_model != existing_path_model:
            raise JudgeResumeConflictError(
                f"Judge model mismatch: requested '{judge_model}', existing run has '{existing_model}'"
            )
    if predictions_sha and existing.get("predictions_sha256"):
        existing_sha = existing["predictions_sha256"]
        if predictions_sha != existing_sha:
            raise JudgeResumeConflictError(
                f"Predictions SHA-256 mismatch: current={predictions_sha[:16]}..., "
                f"existing={existing_sha[:16]}..."
            )

    current_dataset_adapter = dataset_descriptor.name if dataset_descriptor else ""
    current_dataset_version = dataset_descriptor.adapter_version if dataset_descriptor else ""
    _validate_resume_value(
        existing,
        "dataset_adapter",
        current_dataset_adapter,
        "Dataset adapter",
    )
    _validate_resume_value(
        existing,
        "dataset_adapter_version",
        current_dataset_version,
        "Dataset adapter version",
    )
    _validate_resume_value(
        existing,
        "conversion_adapter",
        conversion_adapter,
        "Conversion adapter",
    )
    _validate_resume_value(
        existing,
        "conversion_adapter_version",
        conversion_adapter_version,
        "Conversion adapter version",
    )

    existing_dataset_files = existing.get("dataset_files")
    current_dataset_files = [item.model_dump() for item in dataset_files]
    if existing_dataset_files is None and current_dataset_files:
        raise JudgeResumeConflictError(
            "Dataset digest is missing from the existing run; start a new Judge run"
        )
    if existing_dataset_files is not None and existing_dataset_files != current_dataset_files:
        raise JudgeResumeConflictError(
            "Dataset digest mismatch; the dataset file changed since this Judge run"
        )

    existing_metric_parameters = existing.get("metric_parameters", {}).get(kind.value, {})
    for metric, current in metric_parameters.items():
        previous = existing_metric_parameters.get(metric)
        if previous is None:
            continue
        previous_normalized = MetricRunParameters.model_validate(previous).model_dump()
        if previous_normalized != current.model_dump() and not allow_parameter_change:
            raise JudgeResumeConflictError(
                f"Metric parameter mismatch for {kind.value}.{metric}; "
                "use --force-metrics to recompute that metric"
            )


def _validate_resume_value(
    existing: dict[str, Any],
    field: str,
    current: str,
    label: str,
) -> None:
    previous = existing.get(field)
    if previous in (None, "") and current:
        raise JudgeResumeConflictError(
            f"{label} is missing from the existing run; start a new Judge run"
        )
    if previous not in (None, "") and previous != current:
        raise JudgeResumeConflictError(
            f"{label} mismatch: current={current!r}, existing={previous!r}"
        )


def _indexing_input_contract(
    *,
    framework: str,
    base_path: Path,
    folder_name: str | None,
    project: str | None,
    dataset: str | None,
    source_run_id: str | None,
) -> dict[str, str]:
    """Return the complete deterministic input identity for indexing metrics."""
    return {
        "framework": framework,
        "base_path": str(base_path.resolve()),
        "folder_name": folder_name or "",
        "project": project or "",
        "dataset": dataset or "",
        "source_run_id": source_run_id or "",
    }


def _validate_indexing_resume(
    existing: dict[str, Any],
    current: dict[str, str],
    *,
    allow_input_change: bool,
) -> None:
    """Reject a resume whose indexing source differs from the persisted run."""
    previous = existing.get("indexing_input")
    if previous is None:
        if allow_input_change:
            return
        raise JudgeResumeConflictError(
            "Indexing input contract is missing from the existing run; "
            "resume with --force to replace it"
        )
    if previous != current and not allow_input_change:
        raise JudgeResumeConflictError(
            "Indexing input mismatch; use --force to replace the indexing result"
        )


def _merge_summary(
    layout: Any,
    new_summary: Any,
    kind: EvaluationKind,
    run_id: str,
) -> dict[str, Any]:
    """Merge new evaluation results into the summary.

    Reads the existing summary.json artifact (not the run manifest, which has
    no average_scores). Preserves per-kind metrics/counts without accidental
    overwrite and avoids double-counting the same sample population.
    """
    # Load existing summary artifact if present
    existing: dict[str, Any] = {}
    if layout.summary_file.exists():
        try:
            with open(layout.summary_file, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged_scores: dict[str, float] = {}
    if existing:
        merged_scores.update(existing.get("average_scores", {}))

    if hasattr(new_summary, "average_scores"):
        merged_scores.update(new_summary.average_scores)
    elif isinstance(new_summary, dict):
        merged_scores.update(new_summary.get("average_scores", {}))

    # Track per-kind counts to avoid double-counting
    kind_key = f"kind_{kind.value}"
    existing_kind_counts: dict[str, dict[str, int]] = existing.get("kind_counts", {})
    if kind_key not in existing_kind_counts:
        existing_kind_counts[kind_key] = {
            "total_samples": (
                new_summary.total_samples
                if hasattr(new_summary, "total_samples")
                else new_summary.get("total_samples", 0)
            ),
            "successful_samples": (
                new_summary.successful_samples
                if hasattr(new_summary, "successful_samples")
                else new_summary.get("successful_samples", 0)
            ),
            "failed_samples": (
                new_summary.failed_samples
                if hasattr(new_summary, "failed_samples")
                else new_summary.get("failed_samples", 0)
            ),
        }
    else:
        # Update existing kind counts (e.g. overwrite on re-run with force)
        existing_kind_counts[kind_key]["total_samples"] = (
            new_summary.total_samples
            if hasattr(new_summary, "total_samples")
            else new_summary.get("total_samples", 0)
        )
        existing_kind_counts[kind_key]["successful_samples"] = (
            new_summary.successful_samples
            if hasattr(new_summary, "successful_samples")
            else new_summary.get("successful_samples", 0)
        )
        existing_kind_counts[kind_key]["failed_samples"] = (
            new_summary.failed_samples
            if hasattr(new_summary, "failed_samples")
            else new_summary.get("failed_samples", 0)
        )

    # Merge valid counts
    merged_valid_counts: dict[str, int] = {}
    if existing:
        merged_valid_counts.update(existing.get("metric_valid_counts", {}))
    new_valid = (
        new_summary.metric_valid_counts
        if hasattr(new_summary, "metric_valid_counts")
        else new_summary.get("metric_valid_counts", {})
    )
    merged_valid_counts.update(new_valid)

    return {
        "judge_run_id": run_id,
        "average_scores": merged_scores,
        "metric_valid_counts": merged_valid_counts,
        "kind_counts": existing_kind_counts,
    }


def _validate_persisted_metric_results(
    result_data: dict[str, Any],
    kind: str,
) -> None:
    """Validate detailed and aggregate metrics before writing result JSON."""
    for item in result_data.get("detailed", []):
        if not isinstance(item, dict):
            validate_metric_mapping(
                item,
                scope=f"persisted {kind} detailed entry",
            )
        validate_metric_mapping(
            item.get("metrics", {}),
            scope=(f"persisted {kind} sample id={item.get('id')} seq={item.get('seq')}"),
        )
    validate_metric_mapping(
        result_data.get("average_scores", {}),
        scope=f"persisted {kind} average_scores",
    )


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    if hasattr(summary, "model_dump"):
        return summary.model_dump()
    return dict(summary)


def _execution_model(llm: Any) -> str:
    """Return the model used for this update for manifest auditability."""
    if llm is None:
        return "deterministic"
    return str(getattr(getattr(llm, "config", None), "model", "unknown"))


def _sanitize_base_url(llm: Any) -> str:
    if llm is None:
        return "none"
    config = getattr(llm, "config", None)
    if config is None:
        return "unknown"
    base_url = getattr(config, "base_url", "") or ""
    if not base_url:
        return "default"
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if not parsed.hostname:
        return "default"
    port_part = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port_part}"


def _collect_dataset_digests(gt_repo: Any, dataset: str | None) -> list[FileDigest]:
    """Collect digests for every dataset file used by a Judge evaluation."""
    if dataset is None or gt_repo is None:
        return []
    try:
        ds_file = gt_repo.resolve_dataset_path(dataset)
    except FileNotFoundError:
        return []
    return [
        FileDigest(
            path=str(ds_file),
            sha256=sha256_file(ds_file),
            size_bytes=ds_file.stat().st_size,
        )
    ]


def _conversion_adapter_contract(predictions_file: Path) -> tuple[str, str]:
    """Read the adapter identity recorded beside canonical predictions."""
    manifest_path = predictions_file.parent.parent / "conversion_manifest.json"
    if not manifest_path.is_file():
        return "", ""
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (json.JSONDecodeError, OSError) as exc:
        raise JudgeConfigurationError(
            f"Cannot read conversion manifest for resume contract: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise JudgeConfigurationError(f"Conversion manifest must be an object: {manifest_path}")
    adapter = manifest.get("adapter", "")
    version = manifest.get("adapter_version", "")
    if not isinstance(adapter, str) or not isinstance(version, str):
        raise JudgeConfigurationError(
            f"Conversion manifest has invalid adapter contract: {manifest_path}"
        )
    return adapter, version


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    import subprocess

    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(result.strip())
    except Exception:
        return False
