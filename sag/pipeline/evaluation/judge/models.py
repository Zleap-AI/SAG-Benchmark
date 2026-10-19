"""Judge data models — pure data, no LLM/embedding clients.

Includes predictions schema, conversion/manifest DTOs, and Judge run status enums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvaluationKind(StrEnum):
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    INDEXING = "indexing"


class JudgeRunStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class SampleEvaluationStatus(StrEnum):
    SUCCESS = "success"
    SAMPLE_FAILED = "sample_failed"
    PROGRAMMING_ERROR = "programming_error"


# ---------------------------------------------------------------------------
# Predictions schema (single canonical JudgeSample)
# ---------------------------------------------------------------------------


class JudgeSample(BaseModel):
    """A single evaluation sample — canonical predictions format.

    Compatible input aliases are resolved in a single validator so that
    generation and retrieval code never need to guess field names.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: int = Field(..., description="Sample index (0-based, maps to raw dataset)")
    seq: int | None = Field(
        default=None,
        description="Row index within the predictions file — the unique sample "
        "identity for evaluation storage; distinct from id, because different "
        "source ids can produce the same question (e.g. MusiQue) and thus collide "
        "on the same dataset index.",
    )
    question: str = Field(..., description="Question text")
    answer: str = Field(default="", description="Generated answer")
    ground_truth: str | list[str] = Field(default="", description="Ground truth answer(s)")
    contexts: list[str] = Field(default_factory=list, description="Retrieved contexts")
    evidences: list[str] = Field(default_factory=list, description="Gold evidence statements")
    source: str = Field(default="", description="Dataset name (e.g. hotpotqa)")
    question_type: str = Field(default="qa", description="Question type label")

    @classmethod
    def from_predictions_row(cls, row: dict[str, Any]) -> JudgeSample:
        """Normalise a predictions row with legacy field aliases."""
        if not isinstance(row, dict):
            raise ValueError("prediction row must be an object")
        data: dict[str, Any] = {}
        data["id"] = row["id"]
        data["question"] = row.get("question", "")
        data["answer"] = row["answer"] if "answer" in row else row.get("generated_answer", "")
        data["ground_truth"] = row.get("ground_truth", "")
        data["contexts"] = _as_str_list(row.get("contexts", row.get("context", [])))
        data["evidences"] = _as_str_list(row.get("evidences", row.get("evidence", [])))
        data["source"] = row.get("source", "")
        data["question_type"] = row.get("question_type", "qa")
        return cls(**data)

    @field_validator("question", "answer", "source", "question_type", mode="before")
    @classmethod
    def _validate_text_fields(cls, value: Any, info: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")
        if info.field_name == "question" and not value.strip():
            raise ValueError("question must not be empty")
        return value


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("contexts and evidences must contain only strings")
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value is None:
        return []
    raise ValueError("contexts and evidences must be a string, list of strings, or null")


# ---------------------------------------------------------------------------
# Judge metric / result models
# ---------------------------------------------------------------------------


class JudgeMetricResult(BaseModel):
    """Result for a single metric on a single sample."""

    metric: str = Field(..., description="Metric name")
    score: float = Field(..., description="Finite score value")
    intermediate: dict[str, Any] | None = Field(
        default=None, description="LLM intermediate output (detailed mode)"
    )


class JudgeDetailedResult(BaseModel):
    """Per-sample detailed result (always persisted for auditability)."""

    id: int
    seq: int | None = Field(
        default=None,
        description="Predictions row index — unique sample identity for storage; "
        "keeps duplicate-id samples distinct.",
    )
    question: str
    ground_truth: str | list[str]
    generated_answer: str
    contexts: list[str]
    metrics: dict[str, float]
    llm_intermediate: dict[str, Any] | None = None
    status: SampleEvaluationStatus = SampleEvaluationStatus.SUCCESS
    error_type: str | None = None
    error_message: str | None = None


class JudgeRunSummary(BaseModel):
    """Overall run summary with per-metric averages and detailed entries."""

    average_scores: dict[str, float] = Field(default_factory=dict)
    detailed: list[JudgeDetailedResult] = Field(default_factory=list)
    total_tokens: dict[str, int] = Field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    metric_valid_counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conversion DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    """Request to convert a project's native results into predictions."""

    project: str
    dataset: str
    input_root: Path
    dataset_dir: Path
    artifact_run_root: Path | None = None
    predictions_dir: Path | None = None
    allow_overwrite: bool = True
    mode: str | None = None
    source_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRun:
    """Immutable description of an adapter-located read-only source run."""

    project: str
    dataset: str
    run_root: Path
    artifact_run_root: Path
    source_files: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_root.is_absolute():
            raise ValueError(f"run_root must be absolute: {self.run_root}")


@dataclass(frozen=True, slots=True)
class AdapterConversion:
    """Pure data from adapter — rows plus metadata, no file I/O."""

    rows: tuple[dict[str, Any], ...]
    source_files: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Result of a successful conversion by PredictionConversionService."""

    predictions_path: Path
    manifest_path: Path
    source_run: SourceRun
    row_count: int
    input_digest: str = ""
    output_digest: str = ""


# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------


class FileDigest(BaseModel):
    path: str
    sha256: str
    size_bytes: int


class ConversionManifest(BaseModel):
    schema_version: int = 1
    project: str = ""
    dataset: str = ""
    source_run_root: str = ""
    source_run_id: str = ""
    source_files: list[FileDigest] = Field(default_factory=list)
    predictions_file: str = ""
    row_count: int = 0
    adapter: str = ""
    adapter_version: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeRunParameters(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    max_concurrent: int = 4
    context_top_k: int = 5
    num_samples: int | None = None
    force: bool = False
    force_metrics: bool = False
    retry_failed: bool = False


class MetricRunParameters(BaseModel):
    """Outcome-affecting parameters recorded independently for every metric."""

    context_top_k: int = 5
    num_samples: int | None = None


class JudgeRunManifest(BaseModel):
    schema_version: int = 1
    judge_run_id: str = ""
    status: JudgeRunStatus = JudgeRunStatus.SUCCESS
    started_at: str = ""
    completed_at: str = ""
    judge_model: str = ""
    base_url_display: str = ""
    predictions_path: str = ""
    predictions_sha256: str = ""
    dataset: str = ""
    dataset_adapter: str = ""
    dataset_adapter_version: str = ""
    dataset_files: list[FileDigest] = Field(default_factory=list)
    conversion_adapter: str = ""
    conversion_adapter_version: str = ""
    evaluation_kinds: list[str] = Field(default_factory=list)
    parameters: JudgeRunParameters = Field(default_factory=JudgeRunParameters)
    evaluation_parameters: dict[str, JudgeRunParameters] = Field(default_factory=dict)
    metric_parameters: dict[str, dict[str, MetricRunParameters]] = Field(default_factory=dict)
    indexing_input: dict[str, str] = Field(default_factory=dict)
    kind_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    evidence_dataset_files: list[FileDigest] = Field(default_factory=list)
    git_commit: str = ""
    git_dirty: bool = False
    python_version: str = ""
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    result_files: dict[str, str] = Field(default_factory=dict)
    metric_updates: list[dict[str, Any]] = Field(default_factory=list)


class LatestRunPointer(BaseModel):
    schema_version: int = 1
    latest_run_id: str = ""
    latest_run_dir: str = ""
