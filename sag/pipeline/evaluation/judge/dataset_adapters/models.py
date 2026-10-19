"""Canonical models shared by dataset adapters and the Judge repository."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetCapability(StrEnum):
    """Capabilities that a raw dataset adapter explicitly provides."""

    EVIDENCE_RECALL = "evidence_recall"


class DatasetDescriptor(BaseModel):
    """Stable identity and capabilities declared by one dataset adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[DatasetCapability] = frozenset()
    adapter_version: str = "1.0.0"

    @field_validator("name", "adapter_version")
    @classmethod
    def _require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset descriptor values must not be empty")
        return value

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not alias.strip() for alias in value):
            raise ValueError("dataset aliases must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("dataset aliases must be unique")
        return value


class CanonicalGroundTruthSample(BaseModel):
    """One raw benchmark row in the format consumed by Judge services."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    id: int
    dataset_sample_id: str | None = Field(
        default=None,
        description="Stable sample identifier declared by the raw dataset",
    )
    question: str
    answer: str | list[str] = ""
    evidences: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset", "question")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset and question must not be empty")
        return value

    @field_validator("dataset_sample_id")
    @classmethod
    def _validate_dataset_sample_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("dataset_sample_id must not be empty")
        return value

    @field_validator("evidences")
    @classmethod
    def _validate_evidences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(item, str) for item in value):
            raise ValueError("evidences must contain only strings")
        return value
