"""LightRAG adapter — reads response/<run_id>/hybrid_<ds>_result.json.

Layout precedence: response/<run_id>/ (current, one dir per QA run) →
response/ (legacy flat) → dataset root. Handles empty retrieved_docs by
recording the fact in metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    SourceRunNotFoundError,
)
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)


def _require_dataset_sample_id(row_index: int, item: dict) -> str:
    """Return the source row's non-empty native ``id`` or raise."""
    sample_id = item.get("id", "")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise AdapterConversionError(
            f"LightRAG source row {row_index} is missing a non-empty dataset sample "
            f"id (field 'id'); regenerate the result or migrate it first"
        )
    return sample_id.strip()


def _resolve_result_path(
    ds_dir: Path, dataset: str, source_run_id: str | None = None
) -> Path | None:
    """Locate hybrid_<ds>_result.json, newest run_id first, legacy flat last.

    只有时间戳格式的 run（YYYYMMDD_HHMMSS）参与「取最新」；自定义名
    run（如 --run-id baseline）需用 --source-run-id 显式指定，避免字母
    码点大于数字导致字典序误判为「最新」。无时间戳 run 时回退任意 run。
    """
    filename = f"hybrid_{dataset}_result.json"
    response_root = ds_dir / "response"
    ts_pat = re.compile(r"\d{8}_\d{6}")

    if source_run_id:
        candidate = response_root / source_run_id / filename
        return candidate if candidate.is_file() else None

    if response_root.is_dir():
        run_dirs = [
            e for e in response_root.iterdir() if e.is_dir() and not e.name.startswith("LlmJudge_")
        ]
        ts_runs = sorted((e for e in run_dirs if ts_pat.fullmatch(e.name)), reverse=True)
        for entry in ts_runs:
            candidate = entry / filename
            if candidate.is_file():
                return candidate
        for entry in sorted(run_dirs, reverse=True):
            candidate = entry / filename
            if candidate.is_file():
                return candidate

        flat = response_root / filename
        if flat.is_file():
            return flat

    candidates = sorted(ds_dir.glob("hybrid_*_result.json"), reverse=True)
    return candidates[0] if candidates else None


class LightRAGAdapter:
    """Adapter for LightRAG native results."""

    name = "lightrag"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root
        dataset = request.dataset
        ds_dir = input_root / dataset

        if not ds_dir.is_dir():
            raise SourceRunNotFoundError(f"LightRAG: dataset directory not found: {ds_dir}")

        resp_path = _resolve_result_path(ds_dir, dataset, getattr(request, "source_run_id", None))
        if resp_path is None:
            raise SourceRunNotFoundError(
                f"LightRAG: result file not found under {ds_dir}/response "
                f"(looked for hybrid_{dataset}_result.json)"
            )

        artifact_root = request.artifact_run_root or ds_dir
        source_files = [resp_path]

        # source_run_id mirrors the response/<run_id> directory; fall back to
        # "flat" for the legacy response/<file> or dataset-root layouts.
        parent_name = resp_path.parent.name
        source_run_id = parent_name if parent_name != "response" else "flat"

        return SourceRun(
            project="lightrag",
            dataset=dataset,
            run_root=ds_dir.resolve(),
            artifact_run_root=artifact_root.resolve(),
            source_files=tuple(source_files),
            metadata={
                "result_file": str(resp_path.resolve()),
                "source_run_id": request.source_run_id or source_run_id,
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        dataset_name = source.dataset
        # locate_source already resolved the exact file — reuse it.
        resp_path = (
            source.source_files[0]
            if source.source_files
            else _resolve_result_path(source.run_root, dataset_name)
        )
        if resp_path is None or not Path(resp_path).is_file():
            raise SourceRunNotFoundError(f"LightRAG: result file not found under {source.run_root}")

        with open(resp_path, encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        empty_docs_count = 0
        for row_index, item in enumerate(data):
            docs = item.get("retrieved_docs", [])
            if not docs:
                empty_docs_count += 1
            context_text = item.get("context", "")
            if not context_text and docs:
                context_text = "\n\n".join(str(d) for d in docs)

            rows.append(
                {
                    "question": item["query"],
                    "dataset_sample_id": _require_dataset_sample_id(row_index, item),
                    "context": context_text,
                    "contexts": [context_text] if context_text else [],
                    "generated_answer": item.get("pred_answer", ""),
                }
            )

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={
                "adapter_version": "2.1.0",
                "empty_retrieved_docs_count": empty_docs_count,
            },
        )


# Legacy bare function
def lightrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    adapter = LightRAGAdapter()
    source = SourceRun(
        project="lightrag",
        dataset=dataset_name,
        run_root=Path(run_dir).resolve(),
        artifact_run_root=Path(run_dir).resolve(),
    )
    return list(adapter.convert(source).rows)
