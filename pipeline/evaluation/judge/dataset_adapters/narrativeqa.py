"""Explicit NarrativeQA raw-dataset adapter."""

from __future__ import annotations

from typing import Any

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.common import require_answer, require_question
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetDescriptor,
)


class NarrativeQAAdapter(DatasetAdapter):
    descriptor = DatasetDescriptor(name="narrativeqa")

    def parse_sample(
        self,
        raw: dict[str, Any],
        row_index: int,
    ) -> CanonicalGroundTruthSample:
        return CanonicalGroundTruthSample(
            dataset=self.descriptor.name,
            id=row_index,
            # NarrativeQA has no native id field; use the row index as a
            # stable string so result files backfilled with row-index ids can
            # match via match_dataset_sample_id.
            dataset_sample_id=str(row_index),
            question=require_question(raw, self.descriptor.name, row_index),
            answer=require_answer(raw, self.descriptor.name, row_index),
        )

    def validate_dataset(
        self,
        samples: tuple[CanonicalGroundTruthSample, ...],
    ) -> None:
        return None
