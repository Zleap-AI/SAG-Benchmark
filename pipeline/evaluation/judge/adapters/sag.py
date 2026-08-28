"""SAG2 adapter joins one SAG2 retrieval run with its QA results.

The default source layout is output/<dataset>/sag2/<run_id>/. A run must
contain search_results.json plus one qa_*/qa_results.json so generation and
retrieval evaluation describe the same retrieval output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.evaluation.judge.adapters.base import normalize_question
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    SourceRunNotFoundError,
)
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)

_TIMESTAMP_RUN = re.compile(r"\d{8}_\d{6}")


def _run_candidates(input_root: Path, dataset: str) -> list[Path]:
    """Return SAG2 run directories supported by the input-root conventions."""
    if input_root.is_file():
        return []

    roots = (
        input_root / dataset / "sag2",
        input_root / "sag2",
        input_root,
    )
    candidates: list[Path] = []
    for root in roots:
        if (root / "search_results.json").is_file():
            if root.parent.name == "sag2":
                candidates.append(root)
            continue
        if root.name != "sag2" or not root.is_dir():
            continue
        candidates.extend(
            entry
            for entry in root.iterdir()
            if entry.is_dir() and (entry / "search_results.json").is_file()
        )

    unique: dict[Path, Path] = {}
    for candidate in candidates:
        unique[candidate.resolve()] = candidate.resolve()
    return list(unique.values())


def _latest_qa_file(run_root: Path) -> Path | None:
    """Return the newest default QA output beneath one retrieval run."""
    candidates = [
        entry / "qa_results.json"
        for entry in run_root.iterdir()
        if entry.is_dir() and entry.name.startswith("qa_") and (entry / "qa_results.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.parent.name)


def _question_index(item: dict[str, Any], *, label: str, row_index: int) -> int:
    value = item.get("question_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterConversionError(
            f"SAG2 {label} row {row_index} has invalid question_index: {value!r}"
        )
    return value


class SAGAdapter:
    """Adapter for core SAG2 search_results.json and qa_results.json outputs."""

    name = "sag"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root.resolve()
        candidates = _run_candidates(input_root, request.dataset)
        if request.source_run_id:
            candidates = [
                candidate for candidate in candidates if candidate.name == request.source_run_id
            ]
            if not candidates:
                raise SourceRunNotFoundError(
                    f"SAG2 run not found: {request.source_run_id!r} under {input_root}"
                )

        if not candidates:
            raise SourceRunNotFoundError(
                "SAG2 search result not found. Expected "
                f"{input_root}/<dataset>/sag2/<run_id>/search_results.json"
            )

        candidates_with_qa = [(candidate, _latest_qa_file(candidate)) for candidate in candidates]
        complete_candidates = [
            (candidate, qa_file) for candidate, qa_file in candidates_with_qa if qa_file is not None
        ]
        if not complete_candidates:
            raise SourceRunNotFoundError(
                "SAG2 QA result not found. Run scripts/run_qa_benchmark.py "
                "first so Judge can evaluate generation and retrieval from "
                "the same run."
            )

        selected, qa_file = max(
            complete_candidates,
            key=lambda pair: (
                _TIMESTAMP_RUN.fullmatch(pair[0].name) is not None,
                pair[0].name,
            ),
        )
        search_file = selected / "search_results.json"
        assert qa_file is not None

        return SourceRun(
            project=self.name,
            dataset=request.dataset,
            run_root=selected,
            artifact_run_root=(request.artifact_run_root or input_root).resolve(),
            source_files=(search_file, qa_file),
            metadata={
                "source_run_id": selected.name,
                "strategy": "sag2",
                "search_file": str(search_file),
                "qa_file": str(qa_file),
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        search_file, qa_file = source.source_files
        with search_file.open(encoding="utf-8") as stream:
            search_data = json.load(stream)
        with qa_file.open(encoding="utf-8") as stream:
            qa_data = json.load(stream)

        if not isinstance(search_data, list):
            raise AdapterConversionError("SAG2 search_results.json must contain a JSON list")
        if not isinstance(qa_data, dict) or not isinstance(qa_data.get("per_example"), list):
            raise AdapterConversionError(
                "SAG2 qa_results.json must contain a per_example JSON list"
            )

        qa_by_index: dict[int, dict[str, Any]] = {}
        for row_index, item in enumerate(qa_data["per_example"]):
            if not isinstance(item, dict):
                raise AdapterConversionError(f"SAG2 QA row {row_index} must be an object")
            index = _question_index(item, label="QA", row_index=row_index)
            if index in qa_by_index:
                raise AdapterConversionError(f"SAG2 QA has duplicate question_index {index}")
            qa_by_index[index] = item

        rows: list[dict[str, Any]] = []
        search_indexes: set[int] = set()
        for row_index, item in enumerate(search_data):
            if not isinstance(item, dict):
                raise AdapterConversionError(f"SAG2 search row {row_index} must be an object")
            index = _question_index(item, label="search", row_index=row_index)
            if index in search_indexes:
                raise AdapterConversionError(f"SAG2 search has duplicate question_index {index}")
            search_indexes.add(index)

            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                raise AdapterConversionError(f"SAG2 search row {row_index} has an empty question")
            contexts = item.get("retrieved_docs", [])
            if not isinstance(contexts, list) or not all(
                isinstance(context, str) for context in contexts
            ):
                raise AdapterConversionError(
                    f"SAG2 search row {row_index} retrieved_docs must be a list of strings"
                )

            qa_item = qa_by_index.get(index)
            if qa_item is None:
                raise AdapterConversionError(f"SAG2 QA result is missing question_index {index}")
            qa_question = qa_item.get("question", "")
            if not isinstance(qa_question, str) or (
                qa_question and normalize_question(qa_question) != normalize_question(question)
            ):
                raise AdapterConversionError(
                    f"SAG2 QA/search question mismatch at question_index {index}"
                )
            answer = qa_item.get("predicted_answer", "")
            if not isinstance(answer, str):
                raise AdapterConversionError(
                    f"SAG2 QA row {index} has a non-string predicted_answer"
                )

            rows.append(
                {
                    "canonical_row_id": index - 1,
                    "question": question,
                    "contexts": contexts,
                    "generated_answer": answer,
                }
            )

        unexpected_indexes = sorted(set(qa_by_index) - search_indexes)
        if unexpected_indexes:
            raise AdapterConversionError(
                f"SAG2 QA contains question_index values absent from search results: "
                f"{unexpected_indexes[:5]}"
            )

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={
                "adapter_version": "1.0.0",
                "strategy": "sag2",
                "qa_result_dir": qa_file.parent.name,
            },
        )
