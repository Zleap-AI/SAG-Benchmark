"""HyperRAG adapter — resolves naive/hyper/hyper-lite mode.

Mode is part of source identity; mixed modes raise an error.
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

_SUPPORTED_MODES = ("naive", "hyper", "hyper-lite")


def _require_dataset_sample_id(row_index: int, item: dict) -> str:
    """Return the source row's non-empty native ``id`` or raise."""
    sample_id = item.get("id", "")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise AdapterConversionError(
            f"HyperRAG source row {row_index} is missing a non-empty dataset sample "
            f"id (field 'id'); regenerate the result or migrate it first"
        )
    return sample_id.strip()


class HyperRAGAdapter:
    """Adapter for HyperRAG native results — mode-aware."""

    name = "hyperrag"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root
        dataset = request.dataset
        ds_dir = input_root / dataset

        # Support input_root at multiple levels:
        # - dataset root:  <input_root>/<dataset>/response/<mode>_<ds>_result.json
        # - response root:  <input_root>/<mode>_<ds>_result.json
        # - exact JSON:     <input_root>/ (is a .json file)

        resp_dir = ds_dir / "response"
        resp_dir_alt = input_root / "response"
        source_files: list[Path] = []
        found_modes: list[str] = []

        # Level: exact JSON file
        if input_root.is_file() and input_root.suffix == ".json":
            source_files.append(input_root)
            # Infer mode from filename — check longest first (hyper-lite before hyper)
            for mode in sorted(_SUPPORTED_MODES, key=len, reverse=True):
                if input_root.name.startswith(mode):
                    found_modes.append(mode)
                    break
            if not found_modes:
                found_modes.append("unknown")
            run_root = input_root.parent
            ds_dir_for_meta = run_root

        else:
            # Level: dataset/response/[<run_id>/]<mode>_<ds>_result.json
            candidates: list[tuple[str, Path]] = []
            modes_to_scan = (request.mode,) if request.mode else _SUPPORTED_MODES
            source_run_id = getattr(request, "source_run_id", None)
            for search_dir in (resp_dir, resp_dir_alt):
                if not search_dir.is_dir():
                    continue
                # run_id subdirs first (current layout) — newest wins per mode.
                # 只有时间戳 run（YYYYMMDD_HHMMSS）参与「取最新」；自定义名需
                # 显式 --source-run-id，避免字典序把字母名误判为最新。
                ts_pat = re.compile(r"\d{8}_\d{6}")
                run_dirs = (
                    [search_dir / source_run_id]
                    if source_run_id
                    else [
                        e
                        for e in search_dir.iterdir()
                        if e.is_dir() and not e.name.startswith("LlmJudge_")
                    ]
                )
                if not source_run_id:
                    ts = sorted(
                        (e for e in run_dirs if ts_pat.fullmatch(e.name)),
                        reverse=True,
                    )
                    others = [e for e in sorted(run_dirs, reverse=True) if e not in ts]
                    run_dirs = ts + others
                seen_modes = {m for m, _ in candidates}
                for run_dir in run_dirs:
                    if not run_dir.is_dir():
                        continue
                    for mode in modes_to_scan:
                        if mode in seen_modes:
                            continue
                        candidate = run_dir / f"{mode}_{dataset}_result.json"
                        if candidate.is_file():
                            candidates.append((mode, candidate))
                            seen_modes.add(mode)
                # Legacy flat layout: response/<mode>_<ds>_result.json
                if not source_run_id:
                    for mode in modes_to_scan:
                        if mode in seen_modes:
                            continue
                        candidate = search_dir / f"{mode}_{dataset}_result.json"
                        if candidate.is_file():
                            candidates.append((mode, candidate))
                            seen_modes.add(mode)
            if candidates:
                found_modes = [m for m, _ in candidates]
                source_files = [p for _, p in candidates]
            elif not source_run_id:
                # Level: dataset dir with direct result files
                for search_dir in (ds_dir, input_root):
                    if search_dir.is_dir():
                        for mode in modes_to_scan:
                            for candidate in sorted(
                                search_dir.glob(f"{mode}_*_result.json"), reverse=True
                            ):
                                if not any(p.samefile(candidate) for p in source_files):
                                    found_modes.append(mode)
                                    source_files.append(candidate)

            if not found_modes:
                raise SourceRunNotFoundError(
                    f"HyperRAG: no result files found under {input_root} "
                    f"(looked for: {', '.join(_SUPPORTED_MODES)})"
                )

            ds_dir_for_meta = ds_dir if ds_dir.is_dir() else input_root

        if len(found_modes) > 1 and not getattr(request, "mode", None):
            raise AdapterConversionError(
                f"HyperRAG: multiple modes found ({found_modes}). "
                "Specify a single mode by setting --input-root to the response "
                "directory or the exact result JSON file."
            )

        artifact_root = request.artifact_run_root or ds_dir_for_meta
        selected_file = source_files[0]

        # A nested response/<run_id>/ result uses that real directory name.
        # Flat and direct-file layouts have no run directory, so use the exact
        # selected file stem as a deterministic, collision-free anchor.
        response_dirs = (resp_dir, resp_dir_alt)
        is_nested_run = any(
            selected_file.parent.parent == response_dir for response_dir in response_dirs
        )
        source_run_root = selected_file.parent
        source_run_id = selected_file.parent.name if is_nested_run else selected_file.stem

        return SourceRun(
            project="hyperrag",
            dataset=dataset,
            run_root=source_run_root.resolve(),
            artifact_run_root=artifact_root.resolve(),
            source_files=tuple(source_files),
            metadata={
                "mode": found_modes[0],
                "source_run_id": source_run_id,
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        mode = source.metadata.get("mode", "naive")
        dataset_name = source.dataset

        # Use the exact source file that locate_source discovered
        resp_path = None
        if source.source_files:
            resp_path = source.source_files[0]
        else:
            # Fallback: reconstruct from run_root/response/<mode>_<ds>_result.json
            resp_path = source.run_root / "response" / f"{mode}_{dataset_name}_result.json"

        if not resp_path.is_file():
            raise SourceRunNotFoundError(f"HyperRAG: result file not found: {resp_path}")

        with open(resp_path, encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for row_index, item in enumerate(data):
            rows.append(
                {
                    "question": item["query"],
                    "dataset_sample_id": _require_dataset_sample_id(row_index, item),
                    "context": item.get("context", ""),
                    "contexts": [item.get("context", "")] if item.get("context") else [],
                    "generated_answer": item.get("pred_answer", ""),
                }
            )

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={
                "adapter_version": "2.1.0",
                "mode": mode,
            },
        )


# Legacy bare function (assumes hyper mode)
def hyperrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    """Legacy wrapper — assumes 'hyper' mode."""
    adapter = HyperRAGAdapter()
    source = SourceRun(
        project="hyperrag",
        dataset=dataset_name,
        run_root=Path(run_dir).resolve(),
        artifact_run_root=Path(run_dir).resolve(),
        metadata={"mode": "hyper"},
    )
    return list(adapter.convert(source).rows)
