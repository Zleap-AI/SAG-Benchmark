"""Explicit sample benchmark dataset adapter."""

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


class SampleAdapter(DatasetAdapter):
    """Adapter for the repository's small paragraph-based smoke dataset."""

    descriptor = DatasetDescriptor(
        name="sample",
        capabilities=frozenset({DatasetCapability.EVIDENCE_RECALL}),
        adapter_version="1.2.0",
    )

    def parse_sample(
        self,
        raw: dict[str, Any],
        row_index: int,
    ) -> CanonicalGroundTruthSample:
        dataset = self.descriptor.name
        paragraphs = raw.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise _error(dataset, row_index, "paragraphs", "expected a list")

        evidences: list[str] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                raise _error(
                    dataset, row_index, f"paragraphs[{paragraph_index}]", "expected an object"
                )
            supporting = paragraph.get("is_supporting")
            title = paragraph.get("title")
            text = paragraph.get("text")
            if not isinstance(supporting, bool):
                raise _error(
                    dataset,
                    row_index,
                    f"paragraphs[{paragraph_index}].is_supporting",
                    "expected a boolean",
                )
            if not isinstance(text, str):
                raise _error(
                    dataset,
                    row_index,
                    f"paragraphs[{paragraph_index}].text",
                    "expected a string",
                )
            if supporting:
                if not text.strip():
                    raise _error(
                        dataset,
                        row_index,
                        f"paragraphs[{paragraph_index}].text",
                        "supporting paragraph must not be empty",
                    )
                evidences.append(
                    f"{title}\n{text}" if isinstance(title, str) and title.strip() else text
                )

        dataset_sample_id = raw.get("id")
        if dataset_sample_id is not None and (
            not isinstance(dataset_sample_id, str) or not dataset_sample_id.strip()
        ):
            raise _error(dataset, row_index, "id", "expected a non-empty string")

        return CanonicalGroundTruthSample(
            dataset=dataset,
            id=row_index,
            dataset_sample_id=dataset_sample_id,
            question=require_question(raw, dataset, row_index),
            answer=require_answer(raw, dataset, row_index),
            evidences=tuple(evidences),
        )

    def validate_dataset(
        self,
        samples: tuple[CanonicalGroundTruthSample, ...],
    ) -> None:
        return None
