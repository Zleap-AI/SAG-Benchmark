"""GraphRAG adapter — single-variant canonical layout, dataset-sample-id aligned.

Canonical output layout:
  qa/qa_results.json → per_example[*].{question,dataset_sample_id,predicted_answer,gold_answers}
  response/graphrag_<dataset>_result.json → [{question_index,dataset_sample_id,question,retrieved_docs}]

Aligned exclusively by the dataset-native sample id the producers write back
into every row (see graphrag_benchmark/retrieval.py, run_qa_benchmark.py).
Question text is only used for cross-checking, never for identity.
"""

from __future__ import annotations

import glob
import json
import os
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


def _require_dataset_sample_id(row_index: int, row: dict, source: str) -> str:
    """Return a non-empty dataset_sample_id from a result row or raise."""
    sample_id = row.get("dataset_sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise AdapterConversionError(
            f"GraphRAG {source} row {row_index} is missing a non-empty "
            "'dataset_sample_id'; the result file has not been migrated — "
            "run the backfill script or re-run retrieval/QA"
        )
    return sample_id.strip()


def _has_canonical_data_files(dirpath: str) -> bool:
    """Return whether a directory contains GraphRAG canonical output files."""
    path = Path(dirpath)
    if (path / "qa" / "qa_results.json").is_file():
        return True
    response = path / "response"
    return response.is_dir() and any(response.glob("*result*.json"))


class GraphRAGAdapter:
    """Adapter for Microsoft GraphRAG native results."""

    name = "graphrag"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root.resolve()
        if "caches" in input_root.parts and "outputs" not in input_root.parts:
            raise SourceRunNotFoundError(
                "GraphRAG: legacy caches/... input is unsupported; "
                "migrate results to outputs/<dataset>/<batch> first"
            )
        dataset = request.dataset

        # Try direct dataset directory
        candidates: list[Path] = []
        ds_dir = input_root / dataset
        if ds_dir.is_dir():
            for entry in sorted(ds_dir.iterdir(), reverse=True):
                if entry.is_dir() and not entry.name.startswith("LlmJudge_"):
                    if _has_canonical_data_files(str(entry)):
                        candidates.append(entry)
            if _has_canonical_data_files(str(ds_dir)):
                candidates.append(ds_dir)

        # Try prefix match
        if not candidates:
            for entry in sorted(input_root.iterdir(), reverse=True):
                if (
                    entry.is_dir()
                    and entry.name.startswith(f"{dataset}_")
                    and not entry.name.startswith("LlmJudge_")
                ):
                    if _has_canonical_data_files(str(entry)):
                        candidates.append(entry)

        if not candidates:
            raise SourceRunNotFoundError(
                f"GraphRAG: no valid source run found for dataset '{dataset}' under {input_root}"
            )

        if request.source_run_id:
            explicit = ds_dir / request.source_run_id
            if not explicit.is_dir() or not _has_canonical_data_files(str(explicit)):
                raise SourceRunNotFoundError(f"GraphRAG: source run not found: {explicit}")
            run_root = explicit
        else:
            run_root = candidates[0]
        artifact_root = request.artifact_run_root or run_root

        source_files = _collect_graphrag_source_files(run_root, dataset)

        # source_run_id mirrors the batch directory name — a human-readable
        # anchor; the authoritative lineage (source_run_root + file SHA-256)
        # is recorded in the conversion manifest. When GraphRAG output is laid
        # flat directly under the dataset directory (no per-batch subdirectory),
        # fall back to "flat" so the dataset name is never mistaken for a run id.
        source_run_id = request.source_run_id or (
            run_root.name if run_root.resolve() != ds_dir.resolve() else "flat"
        )

        return SourceRun(
            project="graphrag",
            dataset=dataset,
            run_root=run_root.resolve(),
            artifact_run_root=artifact_root.resolve(),
            source_files=tuple(source_files),
            metadata={
                "source_run_id": source_run_id,
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        run_dir = str(source.run_root)
        dataset_name = source.dataset

        # ---- QA results ----
        pe = []
        qa_path = os.path.join(run_dir, "qa", "qa_results.json")
        if not os.path.exists(qa_path):
            raise AdapterConversionError(
                f"GraphRAG: required canonical qa/qa_results.json not found in {run_dir}"
            )
        with open(qa_path, encoding="utf-8") as f:
            pe = json.load(f).get("per_example", [])

        # ---- retrieval context (dataset-sample-id aligned) ----
        # id -> (context, question); question kept for cross-checking only.
        ctx_by_id: dict[str, tuple[str, str]] = {}
        retrieval_candidates = [
            os.path.join(run_dir, "response", f"graphrag_{dataset_name}_result.json"),
            os.path.join(run_dir, "response", "result.json"),
            *sorted(glob.glob(os.path.join(run_dir, "response", "*result*.json"))),
        ]
        sr_path = next((c for c in retrieval_candidates if os.path.exists(c)), None)
        if not sr_path:
            raise AdapterConversionError(
                f"GraphRAG: no retrieval result file found under {run_dir}/response"
            )
        with open(sr_path, encoding="utf-8") as f:
            for row_index, r in enumerate(json.load(f)):
                sample_id = _require_dataset_sample_id(row_index, r, "retrieval")
                question = r.get("question", "").strip()
                if not question:
                    raise AdapterConversionError(
                        f"GraphRAG retrieval row {row_index} is missing non-empty "
                        f"'question' (dataset_sample_id={sample_id!r})"
                    )
                if sample_id in ctx_by_id:
                    raise AdapterConversionError(
                        f"GraphRAG: duplicate dataset_sample_id in retrieval response: "
                        f"{sample_id!r}"
                    )
                ctx_by_id[sample_id] = (
                    "\n\n".join(r.get("retrieved_docs", []) or []),
                    question,
                )

        # ---- Build rows aligned by dataset_sample_id ----
        rows = []
        seen_qa_ids: set[str] = set()
        for row_index, ex in enumerate(pe):
            sample_id = _require_dataset_sample_id(row_index, ex, "qa")
            if sample_id in seen_qa_ids:
                raise AdapterConversionError(
                    f"GraphRAG: duplicate dataset_sample_id in QA results: {sample_id!r}"
                )
            seen_qa_ids.add(sample_id)
            question = ex.get("question", "").strip()
            if not question:
                raise AdapterConversionError(
                    f"GraphRAG QA row {row_index} is missing non-empty 'question' "
                    f"(dataset_sample_id={sample_id!r})"
                )
            if sample_id not in ctx_by_id:
                raise AdapterConversionError(
                    f"GraphRAG: no retrieval context found for dataset_sample_id: {sample_id!r}"
                )
            ctx, retrieval_question = ctx_by_id[sample_id]
            if retrieval_question != question:
                raise AdapterConversionError(
                    f"GraphRAG: dataset_sample_id {sample_id!r} has inconsistent question "
                    f"between QA and retrieval results"
                )
            rows.append(
                {
                    "dataset_sample_id": sample_id,
                    "question": question,
                    "context": ctx,
                    "contexts": [ctx],
                    "generated_answer": ex.get("predicted_answer", "") or "",
                    "ground_truth": ex.get("gold_answers", []),
                }
            )

        # ---- Cross-check id coverage in both directions ----
        missing_ids = sorted(set(ctx_by_id) - seen_qa_ids)
        if missing_ids:
            raise AdapterConversionError(
                "GraphRAG retrieval/QA dataset_sample_id alignment mismatch: "
                f"retrieval-only ids={missing_ids[:5]}"
            )

        matched_ctx = sum(1 for r in rows if r["context"].strip())
        print(f"    context matched: {matched_ctx}/{len(rows)} (by dataset_sample_id)")

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={"adapter_version": "2.1.0"},
        )


def _collect_graphrag_source_files(run_dir: Path, dataset: str) -> list[Path]:
    files: list[Path] = []
    for candidate in [
        run_dir / "qa" / "qa_results.json",
        run_dir / "response" / f"graphrag_{dataset}_result.json",
        run_dir / "response" / "result.json",
    ]:
        if candidate.is_file():
            files.append(candidate)
    return files


# Legacy bare function for backward compatibility
def graphrag_adapter(run_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    adapter = GraphRAGAdapter()
    source = SourceRun(
        project="graphrag",
        dataset=dataset_name,
        run_root=Path(run_dir).resolve(),
        artifact_run_root=Path(run_dir).resolve(),
    )
    conversion = adapter.convert(source)
    return list(conversion.rows)
