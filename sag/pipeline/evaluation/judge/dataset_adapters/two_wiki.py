"""Explicit 2WikiMultihopQA raw-dataset adapter."""

from __future__ import annotations

from typing import Any

from pipeline.evaluation.judge.dataset_adapters.base import DatasetAdapter
from pipeline.evaluation.judge.dataset_adapters.common import (
    require_answer,
    require_question,
)
from pipeline.evaluation.judge.dataset_adapters.models import (
    CanonicalGroundTruthSample,
    DatasetCapability,
    DatasetDescriptor,
)
from pipeline.evaluation.judge.dataset_adapters.two_wiki_common import (
    resolve_two_wiki_supporting_facts,
)


class TwoWikiAdapter(DatasetAdapter):
    descriptor = DatasetDescriptor(
        name="2wikimultihopqa",
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
        # raw["evidences"] contains relation triples and is intentionally not
        # used as gold evidence for the retrieval metric.
        evidences = resolve_two_wiki_supporting_facts(raw, self.descriptor.name, row_index)
        dataset_sample_id: str | None = raw.get("_id") or None
        return CanonicalGroundTruthSample(
            dataset=self.descriptor.name,
            id=row_index,
            dataset_sample_id=dataset_sample_id,
            question=question,
            answer=answer,
            evidences=evidences,
        )

    def validate_dataset(
        self,
        samples: tuple[CanonicalGroundTruthSample, ...],
    ) -> None:
        return None
