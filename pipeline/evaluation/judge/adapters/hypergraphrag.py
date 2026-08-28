"""HyperGraphRAG adapter — reads response/hybrid_<ds>_result.json.

Excludes resume/error files. Records mode=hybrid and top-k in metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.evaluation.judge.adapters.base import (
    _has_data_files,
)
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    SourceRunNotFoundError,
)
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)


class HyperGraphRAGAdapter:
    """Adapter for HyperGraphRAG native results."""

    name = "hypergraphrag"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root
        dataset = request.dataset
        ds_dir = input_root / dataset

        if not ds_dir.is_dir():
            raise SourceRunNotFoundError(f"HyperGraphRAG: dataset directory not found: {ds_dir}")

        if not _has_data_files(str(ds_dir)):
            raise SourceRunNotFoundError(f"HyperGraphRAG: no data files found in {ds_dir}")

        artifact_root = request.artifact_run_root or ds_dir
        response_root = ds_dir / "response"
        response_runs = (
            sorted(
                (
                    entry
                    for entry in response_root.iterdir()
                    if entry.is_dir() and not entry.name.startswith("LlmJudge_")
                ),
                key=lambda entry: entry.name,
                reverse=True,
            )
            if response_root.is_dir()
            else []
        )
        resp_file = ds_dir / "response" / f"hybrid_{dataset}_result.json"
        if request.source_run_id:
            selected_run = response_root / request.source_run_id
            if (
                not selected_run.is_dir()
                or not (selected_run / f"hybrid_{dataset}_result.json").is_file()
            ):
                raise SourceRunNotFoundError(
                    f"HyperGraphRAG: response run not found: {selected_run}"
                )
        else:
            selected_run = next(
                (
                    entry
                    for entry in response_runs
                    if (entry / f"hybrid_{dataset}_result.json").is_file()
                ),
                None,
            )
        selected_file = (
            selected_run / f"hybrid_{dataset}_result.json"
            if selected_run is not None
            else resp_file
        )

        source_files = [selected_file] if selected_file.is_file() else []
        if not source_files:
            for entry in ds_dir.iterdir():
                if (
                    entry.is_file()
                    and entry.name.startswith("hybrid_")
                    and entry.name.endswith("_result.json")
                ):
                    source_files.append(entry)

        if not source_files:
            raise SourceRunNotFoundError(
                f"HyperGraphRAG: no hybrid result file found for dataset {dataset!r} under {ds_dir}"
            )

        return SourceRun(
            project="hypergraphrag",
            dataset=dataset,
            run_root=ds_dir.resolve(),
            artifact_run_root=artifact_root.resolve(),
            source_files=tuple(source_files),
            metadata={
                "mode": "hybrid",
                "response_run_dir": str(selected_run.resolve()) if selected_run else "",
                "source_run_id": request.source_run_id
                or (selected_run.name if selected_run else "flat"),
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        dataset_name = source.dataset
        response_run_dir = source.metadata.get("response_run_dir", "")
        resp_path = (
            source.source_files[0]
            if source.source_files
            else (
                Path(response_run_dir) / f"hybrid_{dataset_name}_result.json"
                if response_run_dir
                else source.run_root / "response" / f"hybrid_{dataset_name}_result.json"
            )
        )
        if not resp_path.is_file():
            # Try direct file
            candidates = sorted(source.run_root.glob("hybrid_*_result.json"), reverse=True)
            if candidates:
                resp_path = candidates[0]
            else:
                raise SourceRunNotFoundError(f"HyperGraphRAG: not found: {resp_path}")

        with open(resp_path, encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for row_index, item in enumerate(data):
            dataset_sample_id = item.get("id")
            if dataset_sample_id is not None and (
                not isinstance(dataset_sample_id, str) or not dataset_sample_id.strip()
            ):
                raise AdapterConversionError(
                    f"HyperGraphRAG source row {row_index} has an invalid id"
                )
            row = {
                "question": item["query"],
                "context": item.get("context", ""),
                "contexts": [item.get("context", "")] if item.get("context") else [],
                "generated_answer": item.get("pred_answer", ""),
            }
            if dataset_sample_id is not None:
                row["dataset_sample_id"] = dataset_sample_id
            rows.append(row)

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={
                "adapter_version": "2.1.0",
                "mode": source.metadata.get("mode", "hybrid"),
            },
        )


# Legacy bare function
def hypergraphrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    adapter = HyperGraphRAGAdapter()
    source = SourceRun(
        project="hypergraphrag",
        dataset=dataset_name,
        run_root=Path(run_dir).resolve(),
        artifact_run_root=Path(run_dir).resolve(),
    )
    return list(adapter.convert(source).rows)
