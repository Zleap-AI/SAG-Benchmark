"""PredictionConversionService — orchestrates convert flow.

Delegates to AdapterRegistry for source discovery and conversion,
GroundTruthRepository for gold enrichment, and JudgeArtifactStore for persistence.
"""

from __future__ import annotations

import time
from typing import Any

from pipeline.evaluation.judge.adapters.registry import AdapterRegistry
from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
    sha256_file,
    sha256_json,
)
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    PredictionValidationError,
)
from pipeline.evaluation.judge.models import (
    ConversionManifest,
    ConversionRequest,
    ConversionResult,
    FileDigest,
    JudgeSample,
)


class PredictionConversionService:
    """Application service for native-result → predictions conversion."""

    def __init__(
        self,
        registry: AdapterRegistry,
        ground_truth_repository: GroundTruthRepository,
        resolver: ArtifactLayoutResolver,
        store: JudgeArtifactStore,
    ) -> None:
        self._registry = registry
        self._ground_truth = ground_truth_repository
        self._resolver = resolver
        self._store = store

    def convert(self, request: ConversionRequest) -> ConversionResult:
        # 1. Lookup adapter
        adapter = self._registry.get(request.project)

        # 2. Locate source run
        source = adapter.locate_source(request)

        # 3. Convert native results
        conversion = adapter.convert(source)

        # 4. Enrich exclusively from validated canonical ground truth.
        # Source outputs are never trusted for ids, answers, or gold evidence.
        enriched_rows: list[dict[str, Any]] = []
        matched_dataset_rows: dict[int, int] = {}
        for source_row_index, row in enumerate(conversion.rows):
            enriched = dict(row)
            try:
                canonical_row_id = enriched.pop("canonical_row_id", None)
                dataset_sample_id = row.get("dataset_sample_id")
                if canonical_row_id is not None:
                    gt_entry = self._ground_truth.match_canonical_id(
                        request.dataset,
                        canonical_row_id,
                        str(row.get("question", "")),
                    )
                elif isinstance(dataset_sample_id, str) and dataset_sample_id.strip():
                    gt_entry = self._ground_truth.match_dataset_sample_id(
                        request.dataset,
                        dataset_sample_id,
                        str(row.get("question", "")),
                    )
                else:
                    raise ValueError(
                        "row has neither a canonical_row_id nor a non-empty "
                        "dataset_sample_id; the adapter must provide a stable "
                        "sample identity (no question-only fallback)"
                    )
            except Exception as exc:
                raise PredictionValidationError(
                    f"Ground-truth matching failed for dataset {request.dataset!r}, "
                    f"question {row.get('question', '')!r}: {exc}"
                ) from exc

            if gt_entry.id in matched_dataset_rows:
                raise AdapterConversionError(
                    f"{adapter.name}: source rows {matched_dataset_rows[gt_entry.id]} and "
                    f"{source_row_index} map to the same dataset row {gt_entry.id}"
                )
            matched_dataset_rows[gt_entry.id] = source_row_index

            enriched["id"] = gt_entry.id
            enriched["ground_truth"] = gt_entry.answer
            enriched["evidences"] = list(gt_entry.evidences)
            enriched.setdefault("source", request.dataset)
            enriched.setdefault("question_type", "qa")

            enriched_rows.append(enriched)

        # 5. Validate rows via JudgeSample
        samples: list[JudgeSample] = []
        for row in enriched_rows:
            try:
                sample = JudgeSample.from_predictions_row(row)
            except Exception as exc:
                raise PredictionValidationError(f"Row validation failed: {exc}") from exc
            if not sample.question.strip():
                raise PredictionValidationError(f"Empty question for sample id={sample.id}")
            samples.append(sample)

        if not samples:
            raise PredictionValidationError("No valid samples after enrichment")

        # 6. Resolve output layout — mirror the source run into a three-layer
        #    evaluation/<project>/<dataset>/<source_run_id>/ directory. The
        #    absolute source_run_root + source_files SHA-256 remain in the
        #    manifest below; source_run_id is only a human-readable anchor.
        artifact_root = request.artifact_run_root or source.artifact_run_root
        source_run_id = source.metadata.get("source_run_id") or request.source_run_id
        layout = self._resolver.predictions(
            artifact_root,
            request.dataset,
            predictions_dir=request.predictions_dir,
            project=request.project,
            source_run_id=source_run_id,
        )

        # 7. Check overwrite before writing
        if layout.predictions_file.exists() and not request.allow_overwrite:
            raise PredictionValidationError(
                f"Predictions file already exists and overwrite is disabled: "
                f"{layout.predictions_file}"
            )

        # 8. Write predictions
        rows_out = [s.model_dump() for s in samples]
        self._store.write_predictions(layout, rows_out)

        # 9. Compute hashes and write manifest
        input_digest = _compute_rows_digest(conversion.rows)
        output_digest = sha256_file(layout.predictions_file)

        source_file_digests = []
        for sf in source.source_files:
            if sf.is_file():
                source_file_digests.append(
                    FileDigest(
                        path=str(sf),
                        sha256=sha256_file(sf),
                        size_bytes=sf.stat().st_size,
                    )
                )

        manifest = ConversionManifest(
            schema_version=1,
            project=request.project,
            dataset=request.dataset,
            source_run_root=str(source.run_root),
            source_run_id=source_run_id or "",
            source_files=source_file_digests,
            predictions_file=str(layout.predictions_file),
            row_count=len(rows_out),
            adapter=adapter.name,
            adapter_version=conversion.metadata.get("adapter_version", "1.0.0"),
            git_commit=_git_commit(),
            git_dirty=_git_dirty(),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata=conversion.metadata,
        )
        self._store.write_conversion_manifest(layout, manifest.model_dump())

        return ConversionResult(
            predictions_path=layout.predictions_file,
            manifest_path=layout.conversion_manifest_file,
            source_run=source,
            row_count=len(rows_out),
            input_digest=input_digest,
            output_digest=output_digest,
        )


def _compute_rows_digest(rows: tuple[dict[str, Any], ...]) -> str:
    return sha256_json(list(rows))


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
