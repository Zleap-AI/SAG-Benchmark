#!/usr/bin/env python3
"""LLM Judge CLI — thin CLI, delegates to conversion/evaluation services.

Usage:
  uv run --frozen --env-file .env python scripts/run_llm_judge.py --project <p> --dataset <ds> [--input-root <in>] [--output-root <out>]
      # No subcommand → auto: convert then evaluate (generation + retrieval).
  uv run --frozen --env-file .env python scripts/run_llm_judge.py convert ...
  uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate ...  # generation + retrieval + optional indexing, unified --metrics
  uv run --frozen --env-file .env python scripts/run_llm_judge.py indexing ...

Indexing runs only when ``indexing`` is present in ``--metrics`` (e.g.
``--metrics qa_em,indexing``), which also requires ``--framework`` and
``--base-path``. Without it, indexing is skipped by default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from pipeline.core.ai.factory import create_llm_client
from pipeline.evaluation.judge import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
    JudgeError,
    JudgeEvaluationRunner,
    JudgeEvaluationService,
    PredictionConversionService,
)
from pipeline.evaluation.judge.adapters.registry import build_default_registry
from pipeline.evaluation.judge.dataset_adapters.models import DatasetCapability
from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.judge.errors import (
    JudgeConfigurationError,
)
from pipeline.evaluation.judge.generation import ALL_GENERATION_METRICS_SORTED
from pipeline.evaluation.judge.models import ConversionRequest
from pipeline.evaluation.judge.retrieval import ALL_RETRIEVAL_METRICS_SORTED
from pipeline.utils import get_logger

logger = get_logger(__name__)

# Repository root for .env detection
REPO_ROOT = Path(__file__).resolve().parents[1]

# Subdirectory name for a project's native Step-3 results under external/<project>/.
EXTERNAL_OUTPUTS_SUBDIR = "outputs"

# Pseudo-metric token that, when present in ``--metrics``, enables the indexing
# phase. The actual graph metric names are produced dynamically by
# ``calculate_indexing_metrics``, so the CLI only needs this sentinel.
INDEXING_METRIC_TOKEN = "indexing"


def _parse_metric_list(
    raw: str | None,
    *,
    option_name: str,
    supported: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Parse and validate a comma-separated metric selection.

    ``None`` means that the caller requested the evaluator defaults.  An
    explicit list is deduplicated while preserving the user's order; the
    evaluator still applies its question-type routing where applicable.
    """
    if raw is None:
        return None
    metrics = tuple(dict.fromkeys(item.strip().lower() for item in raw.split(",")))
    if not metrics or any(not metric for metric in metrics):
        raise JudgeConfigurationError(f"{option_name} must contain metric names")
    illegal = sorted(set(metrics) - set(supported))
    if illegal:
        supported_text = ", ".join(supported)
        raise JudgeConfigurationError(
            f"Invalid metrics for {option_name}: {illegal}. Supported: {supported_text}"
        )
    return metrics


def _route_metrics(
    metrics: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Split a unified metric list into ``(generation_metrics, retrieval_metrics, run_indexing)``.

    Generation and retrieval families are disjoint, so routing is deterministic
    and the user's order is preserved. The ``indexing`` sentinel toggles the
    indexing phase and is otherwise not a metric. Any other token that belongs
    to neither family raises ``JudgeConfigurationError``.
    """
    generation_set = set(ALL_GENERATION_METRICS_SORTED)
    retrieval_set = set(ALL_RETRIEVAL_METRICS_SORTED)
    run_indexing = INDEXING_METRIC_TOKEN in metrics
    routed = tuple(metric for metric in metrics if metric != INDEXING_METRIC_TOKEN)
    gen_metrics = tuple(metric for metric in routed if metric in generation_set)
    ret_metrics = tuple(metric for metric in routed if metric in retrieval_set)
    illegal = tuple(metric for metric in routed if metric not in generation_set | retrieval_set)
    if illegal:
        supported_text = ", ".join((*ALL_GENERATION_METRICS_SORTED, *ALL_RETRIEVAL_METRICS_SORTED))
        raise JudgeConfigurationError(
            f"Invalid metric(s) for --metrics: {illegal}. Supported: {supported_text} "
            f"(plus '{INDEXING_METRIC_TOKEN}' to enable indexing)"
        )
    return gen_metrics, ret_metrics, run_indexing


def _resolve_metric_option(
    args: argparse.Namespace,
    *,
    option_name: str,
    legacy_option_name: str,
    supported: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Resolve a phase-specific option and its legacy ``--only-metrics`` alias."""
    raw = getattr(args, option_name, None)
    legacy_raw = getattr(args, legacy_option_name, None)
    if raw is not None and legacy_raw is not None:
        raise JudgeConfigurationError(
            f"{option_name} and {legacy_option_name} are mutually exclusive; use {option_name}"
        )
    selected = raw if raw is not None else legacy_raw
    return _parse_metric_list(
        selected,
        option_name=(
            f"--{option_name.replace('_', '-')}"
            if raw is not None
            else f"{legacy_option_name} (legacy)"
        ),
        supported=supported,
    )


def _default_retrieval_metrics(
    ground_truth: GroundTruthRepository,
    dataset: str | None,
) -> tuple[str, ...]:
    """Return retrieval defaults, omitting evidence metrics unsupported by a dataset.

    Explicit ``--metrics evidence_recall`` remains a hard capability error in the
    evaluation service.  This only makes the implicit default safe for datasets
    such as NarrativeQA, which have no gold evidence.
    """
    metrics = ALL_RETRIEVAL_METRICS_SORTED
    if dataset:
        descriptor = ground_truth.descriptor(dataset)
        if DatasetCapability.EVIDENCE_RECALL not in descriptor.capabilities:
            metrics = tuple(metric for metric in metrics if metric != "evidence_recall")
            logger.info(
                "Dataset %s has no evidence capability; skipping default "
                "retrieval metric evidence_recall",
                dataset,
            )
    return metrics


def _new_run_id() -> str:
    """Generate a collision-resistant local run identifier."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_external_output_root(project: str) -> Path:
    """Return the default native-result root for one integrated project."""
    if project == "sag":
        return (REPO_ROOT / "output").resolve()
    return (REPO_ROOT / "external" / project / EXTERNAL_OUTPUTS_SUBDIR).resolve()


def _resolve_source_run_id(
    artifact_root: Path,
    project: str,
    dataset: str,
    source_run_id: str | None,
) -> str | None:
    """Return the source_run_id, or the newest mirrored run directory.

    When ``--source-run-id`` is not given, scan the three-layer mirror for the
    newest ``<run_id>/predictions/predictions_<dataset>.json`` so ``evaluate``
    can pick up the latest converted run.

    Newest is judged by ``convert`` write time (directory mtime). Note this can
    diverge from the newest *source* run selected by an adapter
    (``sorted(..., reverse=True)`` dictionary order on the response/ run dirs).
    Prefer an explicit ``--source-run-id`` when cross-checking an old source run.
    """
    if source_run_id:
        return source_run_id
    layer = artifact_root / "evaluation" / project / dataset
    if not layer.is_dir():
        return None
    candidates = sorted(
        (
            entry
            for entry in layer.iterdir()
            if entry.is_dir() and (entry / "predictions" / f"predictions_{dataset}.json").is_file()
        ),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _resolve_evaluation_data_file(args: argparse.Namespace) -> Path:
    """Resolve the predictions file to evaluate.

    Precedence: explicit ``--data-file``, else ``--artifact-run-root`` (or the
    repository root) combined with ``--project``/``--dataset`` for the
    three-layer mirror, or ``--dataset`` alone for the legacy flat layout.
    """
    data_file = getattr(args, "data_file", None)
    if data_file:
        return Path(data_file).resolve()

    artifact_run_root = getattr(args, "artifact_run_root", None)
    root = Path(artifact_run_root).resolve() if artifact_run_root else REPO_ROOT
    project = getattr(args, "project", None)
    dataset = getattr(args, "dataset", None)

    if project and dataset:
        source_run_id = _resolve_source_run_id(
            root, project, dataset, getattr(args, "source_run_id", None)
        )
        if source_run_id is None:
            raise JudgeConfigurationError(
                f"No converted predictions found under "
                f"{root / 'evaluation' / project / dataset}/; run `convert` first"
            )
        return ArtifactLayoutResolver.predictions(
            root, dataset, project=project, source_run_id=source_run_id
        ).predictions_file

    if dataset:
        return ArtifactLayoutResolver.predictions(root, dataset).predictions_file

    raise JudgeConfigurationError(
        "--dataset (with --project for the three-layer mirror) is required to "
        "resolve predictions when --data-file is not given"
    )


def _print_env_info() -> None:
    env_path = REPO_ROOT / ".env"
    print(f"[env] 根 .env 路径: {env_path}")
    if env_path.exists():
        print("[env] 文件存在")
        has_key = "OPENAI_API_KEY" in os.environ or "API_KEY" in os.environ
        print(f"[env] API key 已配置: {'是' if has_key else '否'}")
    else:
        print("[env] 文件不存在")


def _setup_logging() -> None:
    """Configure logging so judge progress and per-call LLM logs are visible.

    run_llm_judge.py previously had no logging config, so every INFO record
    (runner progress + `pipeline.ai.openai` LLM call/token logs) fell through to
    Python's "last resort" WARNING handler and was hidden. Mirror
    run_qa_benchmark.py: enable INFO at the root handler, then suppress the
    `pipeline` namespace and noisy third-party libs, re-enabling only the loggers
    that carry progress/call info.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress the whole `pipeline` namespace by default…
    logging.getLogger("pipeline").setLevel(logging.WARNING)
    # …then re-enable the specific loggers that carry progress/call info.
    logging.getLogger("pipeline.ai.openai").setLevel(logging.INFO)  # LLM call + token usage
    logging.getLogger("pipeline.ai.llm").setLevel(logging.INFO)  # retry timing
    # `get_logger` prefixes `pipeline.`, and judge modules already live under
    # `pipeline.evaluation.judge`, so their logger names are double-prefixed
    # (e.g. pipeline.pipeline.evaluation.judge.runner).
    logging.getLogger("pipeline.pipeline.evaluation.judge").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)  # this script's own logger
    for _noisy in (
        "httpx",
        "httpcore",
        "openai",
        "elasticsearch",
        "urllib3",
        "asyncio",
        "aiomysql",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


def _print_judge_endpoint(llm) -> None:
    """Print the judge LLM's actual endpoint (host:port) next to `[run] Judge model`.

    `llm` is None in the deterministic-only generation path; guard accordingly.
    """
    config = getattr(llm, "config", None)
    if config is None:
        return
    base_url = getattr(config, "base_url", None) or "default"
    print(f"[run] Judge endpoint: {base_url}")


def _print_metric_scores(
    *,
    artifact_root: Path,
    judge_run_id: str,
    metrics: tuple[str, ...],
    project: str | None,
    dataset: str | None,
    source_run_id: str | None,
) -> None:
    """Print persisted averages for metrics completed by this command."""
    found = ArtifactLayoutResolver.find_judge_run(
        artifact_root,
        judge_run_id,
        project=project,
        dataset=dataset,
        source_run_id=source_run_id,
    )
    if found is None:
        logger.warning("Judge result summary not found for run %s", judge_run_id)
        return
    layout, _manifest = found
    try:
        with open(layout.summary_file, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Unable to read Judge result summary %s: %s", layout.summary_file, exc)
        return
    scores = dict(summary.get("average_scores", {}))
    scores.update(summary.get("indexing_metrics", {}))
    valid_counts = summary.get("metric_valid_counts", {})
    print("  metrics:")
    for metric in metrics:
        value = scores.get(metric)
        display_value = (
            f"{value:.4f}"
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "unavailable"
        )
        valid_count = valid_counts.get(metric)
        suffix = f" (valid samples: {valid_count})" if valid_count is not None else ""
        print(f"    {metric}: {display_value}{suffix}")


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert native results to predictions."""
    registry = build_default_registry()
    resolver = ArtifactLayoutResolver()

    input_root = (
        Path(args.input_root).resolve()
        if args.input_root
        else _default_external_output_root(args.project)
    )
    artifact_root = _resolve_convert_artifact_root(args, input_root)

    datasets: list[str] = args.datasets or []
    if not datasets and input_root.is_dir():
        datasets = sorted(
            d
            for d in os.listdir(input_root)
            if os.path.isdir(input_root / d) and not d.startswith(".")
        )
    if not datasets:
        logger.error("No datasets found.")
        return 2

    dry_run = getattr(args, "dry_run", False)
    for ds in datasets:
        print(f"[{ds}]")
        request = ConversionRequest(
            project=args.project,
            dataset=ds,
            input_root=input_root,
            dataset_dir=Path(args.dataset_dir).resolve(),
            artifact_run_root=artifact_root,
            predictions_dir=(
                Path(args.predictions_dir).resolve()
                if getattr(args, "predictions_dir", None)
                else None
            ),
            allow_overwrite=not dry_run,
            mode=getattr(args, "mode", None),
            source_run_id=getattr(args, "source_run_id", None),
        )

        if dry_run:
            # Run registry lookup + locate_source, print selected sources
            try:
                adapter = registry.get(args.project)
                source = adapter.locate_source(request)
                source_run_id = source.metadata.get("source_run_id")
                layout = resolver.predictions(
                    source.artifact_run_root,
                    ds,
                    predictions_dir=request.predictions_dir,
                    project=args.project,
                    source_run_id=source_run_id,
                )
                print(f"  (dry-run) project: {source.project}")
                print(f"  (dry-run) source run root: {source.run_root}")
                print(f"  (dry-run) artifact run root: {source.artifact_run_root}")
                if source.source_files:
                    print(f"  (dry-run) source files ({len(source.source_files)}):")
                    for sf in source.source_files:
                        print(f"    - {sf}")
                if source.metadata:
                    print(f"  (dry-run) metadata: {source.metadata}")
                print(f"  (dry-run) predictions → {layout.predictions_file}")
                print(f"  (dry-run) manifest    → {layout.conversion_manifest_file}")
                print("  (dry-run) no directories created, no files written")
            except JudgeError as exc:
                logger.error("Dry-run discovery failed for %s: %s", ds, exc)
                return 3
            continue

        service = PredictionConversionService(
            registry=registry,
            ground_truth_repository=GroundTruthRepository(Path(args.dataset_dir).resolve()),
            resolver=resolver,
            store=JudgeArtifactStore(),
        )

        try:
            result = service.convert(request)
            print(f"  predictions: {result.predictions_path} ({result.row_count} rows)")
            print(f"  manifest:    {result.manifest_path}")
        except JudgeError as exc:
            logger.error("Conversion failed for %s: %s", ds, exc)
            return 3
        print()

    print("Done.")
    return 0


def _resolve_convert_artifact_root(args: argparse.Namespace, input_root: Path) -> Path | None:
    """Resolve the canonical artifact root for conversion.

    Defaults to the repository root so predictions are written to the unified
    ``evaluation/<project>/<dataset>/<source_run_id>/`` mirror.
    """
    artifact_root = getattr(args, "artifact_run_root", None)
    predictions_dir = getattr(args, "predictions_dir", None)

    if predictions_dir:
        predictions_path = Path(predictions_dir).resolve()
        if artifact_root:
            root_path = Path(artifact_root).resolve()
            try:
                predictions_path.relative_to(root_path)
            except ValueError as exc:
                raise JudgeConfigurationError(
                    "--predictions-dir must be inside --artifact-run-root"
                ) from exc
        elif predictions_path.name == "predictions":
            if predictions_path.parent.name == "evaluation":
                # legacy flat layout: <root>/evaluation/predictions
                artifact_root = predictions_path.parents[1]
            elif (
                len(predictions_path.parents) >= 4
                and predictions_path.parents[3].name == "evaluation"
            ):
                # three-layer mirror: <root>/evaluation/<project>/<dataset>/<run_id>/predictions
                artifact_root = predictions_path.parents[4]
            else:
                artifact_root = predictions_path.parent
        else:
            artifact_root = predictions_path.parent

    if artifact_root:
        return Path(artifact_root).resolve()
    return REPO_ROOT.resolve()


def _resolve_resume_judge_model(
    artifact_root: Path,
    resume_run_id: str | None,
    project: str | None = None,
    dataset: str | None = None,
    source_run_id: str | None = None,
) -> str | None:
    """Resolve the persisted model directory for a resume run."""
    if not resume_run_id:
        return None
    found = ArtifactLayoutResolver.find_judge_run(
        artifact_root,
        resume_run_id,
        project=project,
        dataset=dataset,
        source_run_id=source_run_id,
    )
    if found is None:
        raise JudgeConfigurationError(
            f"Cannot find Judge run {resume_run_id!r} under {artifact_root}"
        )
    layout, _manifest = found
    return layout.judge_model_dir.name


def _lineage_from_args(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    """Infer ``(project, dataset, source_run_id)`` from the resolved data file.

    Resolving the predictions file first (rather than reading args directly)
    means the latest ``source_run_id`` is auto-detected when only
    ``--project``/``--dataset`` are given, so ``find_judge_run`` always receives
    a complete all-or-nothing lineage instead of a partial ``(project, dataset,
    None)`` that would trip the three-layer path guard.
    """
    predictions_file = _resolve_evaluation_data_file(args)
    return ArtifactLayoutResolver.infer_lineage(predictions_file)


def cmd_indexing(args: argparse.Namespace) -> int:
    _validate_run_options(args)
    run_id = getattr(args, "judge_run_id", None) or _new_run_id()
    judge_model = getattr(args, "judge_model", None) or "indexing"
    resume_run_id = getattr(args, "resume_run_id", None)

    # Indexing does not consume predictions, so lineage comes from explicit
    # --project/--dataset/--source-run-id and --artifact-run-root (default repo root).
    project = getattr(args, "project", None)
    dataset = getattr(args, "dataset", None)
    source_run_id = getattr(args, "source_run_id", None)
    artifact_root = (
        Path(args.artifact_run_root).resolve()
        if getattr(args, "artifact_run_root", None)
        else REPO_ROOT.resolve()
    )
    if project and dataset and not source_run_id:
        source_run_id = _resolve_source_run_id(artifact_root, project, dataset, None)
        if source_run_id is None:
            raise JudgeConfigurationError(
                f"No converted predictions found under "
                f"{artifact_root / 'evaluation' / project / dataset}/; run `convert` first"
            )

    print(f"[run] resolved artifact root: {artifact_root}")
    print(f"[run] Judge model: {judge_model}")
    print(f"[run] Judge run ID: {run_id}")
    if resume_run_id:
        print(f"[run] resume run ID: {resume_run_id}")

    gt_repo = GroundTruthRepository(Path(args.dataset_dir).resolve())
    service = JudgeEvaluationService(
        llm=None,
        runner=JudgeEvaluationRunner(),
        ground_truth_repository=gt_repo,
        resolver=ArtifactLayoutResolver(),
        store=JudgeArtifactStore(),
    )

    try:
        result = service.run_indexing(
            framework=args.framework,
            base_path=Path(args.base_path).resolve(),
            artifact_run_root=artifact_root,
            judge_model=judge_model,
            run_id=run_id,
            folder_name=args.folder_name,
            resume_run_id=resume_run_id,
            force=args.force,
            project=project,
            dataset=dataset,
            source_run_id=source_run_id,
        )
        metrics = result.get("metrics", {})
        if metrics:
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        else:
            print(f"No graph data found for {args.framework} in {args.base_path}")
    except JudgeError as exc:
        logger.error("Indexing failed: %s", exc)
        return 3
    return 0


async def _run_evaluate_core(
    args: argparse.Namespace,
    *,
    requested_metrics: tuple[str, ...] | None,
) -> int:
    """Shared generation + retrieval (+ optional indexing) evaluation core.

    ``requested_metrics`` is the unified list routed by name; ``None`` keeps the
    per-phase evaluator defaults and skips indexing. The ``indexing`` sentinel
    in ``requested_metrics`` toggles the indexing phase.
    """
    _validate_run_options(args)
    artifact_root = _resolve_artifact_root(args)
    resume_run_id = getattr(args, "resume_run_id", None)
    _project, _dataset, _source_run_id = _lineage_from_args(args)
    effective_dataset = args.dataset or _dataset
    resume_judge_model = _resolve_resume_judge_model(
        artifact_root, resume_run_id, _project, _dataset, _source_run_id
    )
    predictions_file = _resolve_evaluation_data_file(args)

    if requested_metrics is not None:
        gen_metrics, ret_metrics, run_indexing = _route_metrics(requested_metrics)
        gt_repo = GroundTruthRepository(Path(args.dataset_dir).resolve())
    else:
        gen_metrics = ALL_GENERATION_METRICS_SORTED
        run_indexing = False
        gt_repo = GroundTruthRepository(Path(args.dataset_dir).resolve())
        ret_metrics = _default_retrieval_metrics(gt_repo, effective_dataset)

    # Fail fast on missing indexing inputs before any LLM work is spent.
    if run_indexing and (
        not getattr(args, "framework", None) or not getattr(args, "base_path", None)
    ):
        raise JudgeConfigurationError(
            "the `indexing` metric requires both --framework and --base-path"
        )

    deterministic_metrics = {"qa_em", "qa_f1", "rouge_score"}
    needs_llm = bool(ret_metrics) or any(
        metric not in deterministic_metrics for metric in gen_metrics
    )
    llm = await create_llm_client(scenario="judge") if needs_llm else None

    try:
        execution_model = (
            getattr(getattr(llm, "config", None), "model", "deterministic")
            if llm
            else "deterministic"
        )
        judge_model = resume_judge_model or execution_model
        run_id = getattr(args, "judge_run_id", None) or _new_run_id()
        # Only one run_id exists per command invocation, so generation and
        # retrieval share it. A resume run keeps its own id for both kinds.
        ret_run_id = resume_run_id or run_id

        print(f"[run] resolved artifact root: {artifact_root}")
        print(f"[run] predictions: {predictions_file}")
        print(f"[run] Judge model: {judge_model}")
        _print_judge_endpoint(llm)
        print(f"[run] Judge run ID: {run_id}")
        if resume_run_id:
            print(f"[run] resume run ID: {resume_run_id}")

        service = JudgeEvaluationService(
            llm=llm,
            runner=JudgeEvaluationRunner(max_concurrent=args.max_concurrent),
            ground_truth_repository=gt_repo,
            resolver=ArtifactLayoutResolver(),
            store=JudgeArtifactStore(),
        )

        manifest = None
        if gen_metrics:
            manifest = await service.run_generation(
                predictions_file=predictions_file,
                artifact_run_root=artifact_root,
                judge_model=judge_model,
                run_id=run_id,
                metrics=gen_metrics,
                context_top_k=args.context_top_k,
                num_samples=args.num_samples,
                force=args.force,
                force_metrics=getattr(args, "force_metrics", False),
                retry_failed=getattr(args, "retry_failed", False),
                dataset=effective_dataset,
                resume_run_id=resume_run_id,
            )
            print(f"Generation done: {manifest.status.value}")

        if ret_metrics:
            manifest = await service.run_retrieval(
                predictions_file=predictions_file,
                artifact_run_root=artifact_root,
                judge_model=judge_model,
                run_id=ret_run_id,
                metrics=ret_metrics,
                context_top_k=args.context_top_k,
                num_samples=args.num_samples,
                force=args.force,
                force_metrics=getattr(args, "force_metrics", False),
                retry_failed=getattr(args, "retry_failed", False),
                dataset=effective_dataset,
                resume_run_id=manifest.judge_run_id if manifest else resume_run_id,
            )
            print(f"Retrieval done: {manifest.status.value}")

        completed_metrics = tuple(dict.fromkeys((*gen_metrics, *ret_metrics)))

        # Optional indexing, toggled by the ``indexing`` sentinel in --metrics.
        if run_indexing:
            project, dataset_id, source_run_id = ArtifactLayoutResolver.infer_lineage(
                predictions_file
            )
            if manifest is not None:
                # Append indexing to the generation/retrieval run just created.
                indexing_run_id = manifest.judge_run_id
                indexing_judge_model = manifest.judge_model
                indexing_resume_run_id = manifest.judge_run_id
            elif resume_run_id:
                # Indexing-only resume: append to the persisted run instead of
                # silently starting a new one under the "indexing" model dir.
                indexing_run_id = resume_run_id
                indexing_judge_model = resume_judge_model or "indexing"
                indexing_resume_run_id = resume_run_id
            else:
                # Indexing-only: mirror the standalone `indexing` subcommand so
                # results land in the same model dir.
                indexing_run_id = run_id
                indexing_judge_model = "indexing"
                indexing_resume_run_id = None
            indexing_result = service.run_indexing(
                framework=args.framework,
                base_path=Path(args.base_path).resolve(),
                artifact_run_root=artifact_root,
                judge_model=indexing_judge_model,
                run_id=indexing_run_id,
                folder_name=getattr(args, "folder_name", None),
                resume_run_id=indexing_resume_run_id,
                force=args.force,
                project=project,
                dataset=dataset_id,
                source_run_id=source_run_id,
            )
            completed_metrics = tuple(
                dict.fromkeys((*completed_metrics, *indexing_result.get("metrics", {})))
            )
            print("Indexing done")

        if manifest is not None:
            print(f"Judge run complete: {manifest.judge_run_id}")
            print(f"  status: {manifest.status.value}")
            print(f"  successful: {manifest.successful_samples}/{manifest.total_samples}")
        if run_indexing and manifest is None:
            # Indexing-only: summary lives under the indexing run just written;
            # read scores directly instead of routing through find_judge_run.
            for metric, value in indexing_result.get("metrics", {}).items():
                print(f"  {metric}: {value:.4f}")
        if manifest is not None:
            _print_metric_scores(
                artifact_root=artifact_root,
                judge_run_id=manifest.judge_run_id,
                metrics=completed_metrics,
                project=_project,
                dataset=_dataset,
                source_run_id=_source_run_id,
            )
    finally:
        if llm is not None:
            await llm.close()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    requested_metrics = _resolve_metric_option(
        args,
        option_name="metrics",
        legacy_option_name="only_metrics",
        supported=(
            *ALL_GENERATION_METRICS_SORTED,
            *ALL_RETRIEVAL_METRICS_SORTED,
            INDEXING_METRIC_TOKEN,
        ),
    )
    return asyncio.run(_run_evaluate_core(args, requested_metrics=requested_metrics))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

# 与 DatasetLoader.DEFAULT_DATASET_DIR 保持一致：仓库根下的 dataset/
DEFAULT_DATASET_DIR = str(Path(__file__).resolve().parent.parent / "dataset")


def _add_auto_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared options for the no-subcommand auto mode (convert → evaluate)."""
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--input-root",
        default=None,
        help="Input root for native results (default: external/<project>/outputs; SAG: output/)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Artifact root for the evaluation mirror and results (default: repository root)",
    )
    parser.add_argument("--artifact-run-root", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--judge-run-id", default=None)
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--context-top-k", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="num_samples")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="源数据集目录")
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--mode", default=None, choices=["naive", "hyper", "hyper-lite"])
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument(
        "--metrics",
        default=None,
        metavar="METRICS",
        help="逗号分隔的评测指标；含 indexing 时触发索引阶段（需 --framework 和 --base-path）",
    )
    parser.add_argument(
        "--only-metrics",
        default=None,
        metavar="METRICS",
        help="兼容旧参数；请改用 --metrics",
    )
    parser.add_argument("--framework", default=None)
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--folder-name", default=None)


def _normalize_auto_args(args: argparse.Namespace) -> None:
    """Validate and normalise the no-subcommand auto-mode arguments."""
    project = getattr(args, "project", None)
    if not project:
        raise JudgeConfigurationError("--project is required in auto mode (convert + evaluate)")
    datasets = args.datasets or ([args.dataset] if getattr(args, "dataset", None) else [])
    if not datasets:
        raise JudgeConfigurationError(
            "At least one dataset is required in auto mode: --dataset/--datasets"
        )
    args.datasets = datasets
    # Auto mode does not support resume, so it must never request failed-only retries.
    args.retry_failed = False

    output_root = getattr(args, "output_root", None)
    artifact_run_root = getattr(args, "artifact_run_root", None)
    if output_root and artifact_run_root:
        raise JudgeConfigurationError(
            "--output-root and --artifact-run-root are mutually exclusive"
        )
    if output_root:
        args.artifact_run_root = output_root

    if getattr(args, "resume_run_id", None) or getattr(args, "judge_run_id", None):
        raise JudgeConfigurationError(
            "--resume-run-id/--judge-run-id are not supported in auto mode; "
            "run the child subcommand explicitly for resume flows"
        )


def cmd_auto(args: argparse.Namespace) -> int:
    """No-subcommand flow: convert native results, then evaluate all datasets."""
    _normalize_auto_args(args)
    rc = cmd_convert(args)
    if rc != 0:
        return rc
    if getattr(args, "dry_run", False):
        return 0
    # A single --base-path would recompute the same graph once per dataset.
    if (
        len(args.datasets) > 1
        and (args.metrics or args.only_metrics)
        and INDEXING_METRIC_TOKEN in (args.metrics or args.only_metrics or "").lower()
    ):
        logger.warning(
            "auto mode with multiple datasets runs indexing once per dataset "
            "against the same --base-path; prefer the `indexing` subcommand"
        )
    for ds in args.datasets:
        ds_args = argparse.Namespace(**vars(args))
        ds_args.dataset = ds
        rc = cmd_evaluate(ds_args)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Judge evaluation CLI")
    parser.set_defaults(command=None)
    _add_auto_mode_arguments(parser)
    sub = parser.add_subparsers(dest="command")

    # convert
    p_conv = sub.add_parser("convert", help="Convert method outputs to predictions")
    p_conv.add_argument("--project", required=True)
    p_conv.add_argument(
        "--input-root",
        default=None,
        help="Input root for native results (default: external/<project>/outputs; SAG: output/)",
    )
    p_conv.add_argument("--datasets", nargs="*", default=None)
    p_conv.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET_DIR, help="源数据集目录（默认：仓库根 dataset/）"
    )
    p_conv.add_argument("--artifact-run-root", default=None)
    p_conv.add_argument(
        "--predictions-dir",
        default=None,
        help="Advanced: override predictions output directory",
    )
    p_conv.add_argument("--dry-run", action="store_true")
    p_conv.add_argument(
        "--mode",
        choices=["naive", "hyper", "hyper-lite"],
        default=None,
        help="Hyper-RAG source mode; required when multiple modes exist",
    )
    p_conv.add_argument("--source-run-id", default=None, help="Select a timestamped source run")

    # evaluate (generation + retrieval, metrics routed by name)
    p_eval = sub.add_parser("evaluate", help="Run generation and/or retrieval evaluation")
    p_eval.add_argument("--data-file", default=None)
    p_eval.add_argument("--project", default=None)
    p_eval.add_argument("--artifact-run-root", default=None)
    p_eval.add_argument("--judge-run-id", default=None)
    p_eval.add_argument(
        "--resume-run-id",
        default=None,
        help="Resume a previous run (mutually exclusive with --judge-run-id)",
    )
    p_eval.add_argument("--max-concurrent", type=int, default=3)
    p_eval.add_argument(
        "--metrics",
        default=None,
        metavar="METRICS",
        help=(
            "逗号分隔的评测指标；按名称自动路由到对应阶段。generation 支持 "
            + ", ".join(ALL_GENERATION_METRICS_SORTED)
            + "；retrieval 支持 "
            + ", ".join(ALL_RETRIEVAL_METRICS_SORTED)
        ),
    )
    p_eval.add_argument(
        "--only-metrics",
        default=None,
        metavar="METRICS",
        help="兼容旧参数；请改用 --metrics",
    )
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume-run-id, retry failed samples only",
    )
    p_eval.add_argument(
        "--force-metrics",
        action="store_true",
        help=("With --resume-run-id, recompute and replace only the selected metrics"),
    )
    p_eval.add_argument("--dataset", default=None)
    p_eval.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p_eval.add_argument("--source-run-id", default=None)
    p_eval.add_argument("--context-top-k", type=int, default=5)
    p_eval.add_argument("--num-samples", type=int, default=None)
    p_eval.add_argument("--top-k", type=int, default=None, dest="num_samples")
    p_eval.add_argument("--framework", default=None)
    p_eval.add_argument("--base-path", default=None)
    p_eval.add_argument("--folder-name", default=None)

    # indexing
    p_idx = sub.add_parser("indexing", help="Run graph indexing metrics")
    p_idx.add_argument("--framework", required=True)
    p_idx.add_argument("--base-path", required=True)
    p_idx.add_argument("--folder-name", default=None)
    p_idx.add_argument("--project", default=None)
    p_idx.add_argument("--dataset", default=None)
    p_idx.add_argument("--source-run-id", default=None)
    p_idx.add_argument("--artifact-run-root", default=None)
    p_idx.add_argument("--judge-run-id", default=None)
    p_idx.add_argument("--resume-run-id", default=None)
    p_idx.add_argument("--judge-model", default=None)
    p_idx.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p_idx.add_argument("--force", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Run option validation
# ---------------------------------------------------------------------------


def _validate_run_options(args: argparse.Namespace) -> None:
    """Validate non-overwriting run and metric-level overwrite options."""
    resume_run_id = getattr(args, "resume_run_id", None)
    judge_run_id = getattr(args, "judge_run_id", None)
    if resume_run_id and judge_run_id:
        raise JudgeConfigurationError(
            "--resume-run-id and --judge-run-id are mutually exclusive. "
            "Use --resume-run-id to resume an existing run, or "
            "--judge-run-id to create a new run with a specific ID."
        )
    force_metrics = getattr(args, "force_metrics", False)
    retry_failed = getattr(args, "retry_failed", False)
    force = getattr(args, "force", False)
    if retry_failed and not resume_run_id:
        raise JudgeConfigurationError("--retry-failed requires --resume-run-id")
    if retry_failed and (force or force_metrics):
        raise JudgeConfigurationError(
            "--retry-failed is mutually exclusive with --force and --force-metrics"
        )
    if force_metrics and not resume_run_id:
        raise JudgeConfigurationError("--force-metrics requires --resume-run-id")
    if force_metrics and force:
        raise JudgeConfigurationError("--force and --force-metrics are mutually exclusive")
    has_metric_selection = any(
        getattr(args, name, None) is not None for name in ("only_metrics", "metrics")
    )
    if force_metrics and not has_metric_selection:
        raise JudgeConfigurationError("--force-metrics requires --only-metrics or --metrics")
    # `--force-metrics` only makes sense with at least one real metric to
    # recompute; a lone `indexing` token has no per-sample metric to replace.
    if force_metrics:
        selection = getattr(args, "metrics", None) or getattr(args, "only_metrics", None) or ""
        real_metrics = [
            m for m in selection.split(",") if m.strip().lower() != INDEXING_METRIC_TOKEN
        ]
        if not real_metrics:
            raise JudgeConfigurationError(
                "--force-metrics requires at least one real metric; "
                "the `indexing` token alone cannot be force-recomputed"
            )


def _resolve_artifact_root(args: argparse.Namespace) -> Path:
    """Resolve artifact_run_root from args or predictions file layout.

    Requires --artifact-run-root or a valid ArtifactLayoutResolver.infer_run_root
    result. Does NOT silently fall back to predictions_file.parent.
    """
    if getattr(args, "artifact_run_root", None):
        return Path(args.artifact_run_root).resolve()
    predictions_file = _resolve_evaluation_data_file(args)
    try:
        return ArtifactLayoutResolver.infer_run_root(predictions_file)
    except Exception as exc:
        raise JudgeConfigurationError(
            f"Cannot determine artifact run root. "
            f"Provide --artifact-run-root explicitly, or ensure the predictions "
            f"file is under evaluation/predictions/predictions_*.json (or the "
            f"three-layer evaluation/<project>/<dataset>/<run_id>/predictions/). "
            f"(infer_run_root failed: {exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()
    _print_env_info()

    parser = build_parser()
    args = parser.parse_args()

    # Print key parameters
    print(f"[run] command={args.command}")
    if hasattr(args, "project"):
        print(f"[run] project={args.project}")
    if hasattr(args, "data_file"):
        print(f"[run] predictions={args.data_file}")
    if hasattr(args, "max_concurrent"):
        print(f"[run] max_concurrent={args.max_concurrent}")

    commands = {
        "convert": cmd_convert,
        "evaluate": cmd_evaluate,
        "indexing": cmd_indexing,
    }

    try:
        if args.command is None:
            return cmd_auto(args)
        return commands[args.command](args)
    except JudgeError as exc:
        logger.error("Judge error: %s", exc)
        return 3
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        return 10


if __name__ == "__main__":
    sys.exit(main())
