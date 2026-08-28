"""Explicit MuSiQue raw-dataset adapter."""

from __future__ import annotations

from typing import Any

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.common import (
    _error,
    require_answer,
    require_question,
)
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetCapability,
    DatasetDescriptor,
)


class MusiqueAdapter(DatasetAdapter):
    descriptor = DatasetDescriptor(
        name="musique",
        capabilities=frozenset({DatasetCapability.EVIDENCE_RECALL}),
        adapter_version="1.2.0",
    )

    def parse_sample(
        self,
        raw: dict[str, Any],
        row_index: int,
    ) -> CanonicalGroundTruthSample:
        question = require_question(raw, self.descriptor.name, row_index)
        answer = require_answer(raw, self.descriptor.name, row_index)
        dataset_sample_id = raw.get("id")
        if dataset_sample_id is not None and (
            not isinstance(dataset_sample_id, str) or not dataset_sample_id.strip()
        ):
            raise _error(self.descriptor.name, row_index, "id", "expected a non-empty string")
        paragraphs = raw.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise _error(self.descriptor.name, row_index, "paragraphs", "expected a list")

        evidences: list[str] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                raise _error(
                    self.descriptor.name,
                    row_index,
                    f"paragraphs[{paragraph_index}]",
                    "expected an object",
                )
            supporting = paragraph.get("is_supporting")
            text = paragraph.get("paragraph_text")
            if not isinstance(supporting, bool):
                raise _error(
                    self.descriptor.name,
                    row_index,
                    f"paragraphs[{paragraph_index}].is_supporting",
                    "expected a boolean",
                )
            if not isinstance(text, str):
                raise _error(
                    self.descriptor.name,
                    row_index,
                    f"paragraphs[{paragraph_index}].paragraph_text",
                    "expected a string",
                )
            if supporting:
                if not text.strip():
                    raise _error(
                        self.descriptor.name,
                        row_index,
                        f"paragraphs[{paragraph_index}].paragraph_text",
                        "supporting paragraph must not be empty",
                    )
                evidences.append(text.strip())

        return CanonicalGroundTruthSample(
            dataset=self.descriptor.name,
            id=row_index,
            dataset_sample_id=dataset_sample_id,
            question=question,
            answer=answer,
            evidences=tuple(evidences),
        )

    def validate_dataset(
        self,
        samples: tuple[CanonicalGroundTruthSample, ...],
    ) -> None:
        return None
