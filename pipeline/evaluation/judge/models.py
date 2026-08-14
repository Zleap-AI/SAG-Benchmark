"""Judge data models — pure data, no LLM/embedding clients."""

from typing import Any

from pydantic import BaseModel, Field


class JudgeSample(BaseModel):
    """A single evaluation sample.

    Compatible with `predictions_<dataset>.json` format.
    """

    id: int = Field(..., description="Sample index (0-based, maps to raw dataset)")
    question: str = Field(..., description="Question text")
    answer: str = Field(default="", description="Generated answer")
    ground_truth: str | list[str] = Field(
        default="", description="Ground truth answer(s)"
    )
    contexts: list[str] = Field(default_factory=list, description="Retrieved contexts")
    evidences: list[str] = Field(
        default_factory=list, description="Gold evidence statements"
    )
    source: str = Field(default="", description="Dataset name (e.g. hotpotqa)")


class JudgeMetricResult(BaseModel):
    """Result for a single metric on a single sample."""

    metric: str = Field(..., description="Metric name")
    score: float = Field(..., description="Score value (NaN if failed)")
    intermediate: dict[str, Any] | None = Field(
        default=None, description="LLM intermediate output (detailed mode)"
    )


class JudgeDetailedResult(BaseModel):
    """Per-sample detailed result (always persisted for auditability)."""

    id: int
    question: str
    ground_truth: str | list[str]
    generated_answer: str
    contexts: list[str]
    metrics: dict[str, float]
    llm_intermediate: dict[str, Any] | None = None


class JudgeRunSummary(BaseModel):
    """Overall run summary with per-metric averages and detailed entries."""

    average_scores: dict[str, float] = Field(default_factory=dict)
    detailed: list[JudgeDetailedResult] = Field(default_factory=list)
    total_tokens: dict[str, int] = Field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
