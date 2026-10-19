"""HippoRAG2 adapter — reads <llm>_<emb>/qa_result/qa_results_latest.json.

Artifact root binds to a specific model combination directory to prevent
cross-embedding mixing.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    SourceRunAmbiguousError,
    SourceRunNotFoundError,
)
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionRequest,
    SourceRun,
)


class HippoRAG2Adapter:
    """Adapter for the canonical outputs/.../qa_result layout."""

    name = "hipporag2"

    def locate_source(self, request: ConversionRequest) -> SourceRun:
        input_root = request.input_root.resolve()
        if "caches" in input_root.parts and "outputs" not in input_root.parts:
            raise SourceRunNotFoundError(
                "HippoRAG2: legacy caches/... input is unsupported; "
                "migrate results to outputs/.../qa_result first"
            )
        dataset = request.dataset
        ds_dir = input_root / dataset

        # Support input_root at four levels under the canonical outputs root:
        # 1. outputs root: input_root/<dataset>/<llm>_<emb>/qa_result
        # 2. dataset root: input_root/<dataset>/<llm>_<emb>/qa_result
        #                     or input_root/<dataset>/qa_result
        # 3. exact model root: input_root/<llm>_<emb>/qa_result
        # 4. exact qa_result root: input_root/qa_results_latest.json
        candidates: list[Path] = []

        # Level 1: input_root/<dataset>/<llm>_<emb>/qa_result (outputs root)
        if ds_dir.is_dir():
            for entry in sorted(ds_dir.iterdir()):
                if entry.is_dir() and (entry / "qa_result").is_dir():
                    candidates.append(entry)

        # Level 2: input_root/<dataset>/qa_result (dataset = model dir)
        if (ds_dir / "qa_result").is_dir():
            candidates.append(ds_dir)

        # Level 2b: input_root/<dataset> itself is a model dir with qa_result
        # (when dataset matches a subdir name pattern like gpt-4_text-emb)
        for entry in sorted(input_root.iterdir()):
            if entry.is_dir() and (entry / "qa_result").is_dir():
                if entry.name == dataset or "_" in entry.name:
                    candidates.append(entry)

        # Level 3: exact model root — input_root/qa_result exists
        if (input_root / "qa_result").is_dir():
            candidates.append(input_root)

        # Level 4: input_root is itself a qa_result dir
        if (input_root / "qa_results_latest.json").is_file():
            candidates.append(input_root)

        if not candidates:
            raise SourceRunNotFoundError(
                f"HippoRAG2: no <llm>_<emb>/qa_result found at any level under {input_root}"
            )

        # Deduplicate
        seen: set[str] = set()
        unique_candidates: list[Path] = []
        for c in candidates:
            if str(c.resolve()) not in seen:
                seen.add(str(c.resolve()))
                unique_candidates.append(c)

        # A HippoRAG2 source run is the selected model-combination directory
        # (or the explicitly supplied qa_result directory). Apply the selector
        # before the ambiguity check, then persist the selected directory name.
        if request.source_run_id:
            unique_candidates = [
                candidate
                for candidate in unique_candidates
                if candidate.name == request.source_run_id
            ]
            if not unique_candidates:
                raise SourceRunNotFoundError(
                    f"HippoRAG2: source run not found: {request.source_run_id!r} under {input_root}"
                )

        if len(unique_candidates) > 1:
            names = [c.name for c in unique_candidates]
            raise SourceRunAmbiguousError(
                f"HippoRAG2: multiple model directories found: {names}. "
                "Please specify a more specific --input-root to select one "
                "(e.g. .../<llm>_<emb> or .../<llm>_<emb>/qa_result)."
            )

        run_root = unique_candidates[0]
        qa_result_dir = run_root / "qa_result" if (run_root / "qa_result").is_dir() else run_root
        artifact_root = request.artifact_run_root or qa_result_dir

        # The selected run root is a model-combination directory or an exact
        # qa_result directory, so its basename is the reproducible source ID.
        source_run_id = run_root.name

        source_files = []
        for fname in ["qa_results_latest.json", "retrieval_results_latest.json"]:
            fp = qa_result_dir / fname
            if fp.is_file():
                source_files.append(fp)

        return SourceRun(
            project="hipporag2",
            dataset=dataset,
            run_root=qa_result_dir.resolve(),
            artifact_run_root=artifact_root.resolve(),
            source_files=tuple(source_files),
            metadata={
                "model_dir": source_run_id,
                "source_run_id": source_run_id,
            },
        )

    def convert(self, source: SourceRun) -> AdapterConversion:
        qa_file = source.run_root / "qa_results_latest.json"
        if not qa_file.is_file():
            raise SourceRunNotFoundError(
                f"HippoRAG2: qa_results_latest.json not found in {source.run_root}"
            )

        with open(qa_file, encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        seen_ids: set[str] = set()
        for row_index, item in enumerate(data.get("results", [])):
            sample_id = item.get("dataset_sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise AdapterConversionError(
                    f"HippoRAG2 source row {row_index} is missing a non-empty "
                    "'dataset_sample_id'; the result file has not been migrated — "
                    "run the backfill script or re-run retrieval/QA"
                )
            sample_id = sample_id.strip()
            if sample_id in seen_ids:
                raise AdapterConversionError(
                    f"HippoRAG2: duplicate dataset_sample_id in results: {sample_id!r}"
                )
            seen_ids.add(sample_id)
            docs = item.get("docs", [])
            if isinstance(docs, list):
                context_text = "\n\n".join(str(d) for d in docs)
            else:
                context_text = str(docs)
            rows.append(
                {
                    "dataset_sample_id": sample_id,
                    "question": item["question"],
                    "context": context_text,
                    "contexts": docs if isinstance(docs, list) else [],
                    "generated_answer": item.get("answer", ""),
                    "ground_truth": item.get("gold_answers", []),
                }
            )

        return AdapterConversion(
            rows=tuple(rows),
            source_files=source.source_files,
            metadata={
                "adapter_version": "2.1.0",
                "model_dir": source.metadata.get("model_dir", ""),
            },
        )
