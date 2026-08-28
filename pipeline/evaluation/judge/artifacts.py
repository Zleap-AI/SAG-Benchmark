"""Judge artifact layout, atomic storage, hashing, and latest-pointer management.

Pure path strategies — no directory creation, no file I/O beyond what
checkpoint.atomic_write_json already provides.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.evaluation.judge.checkpoint import atomic_write_json
from pipeline.evaluation.judge.errors import ArtifactPathError

# ---------------------------------------------------------------------------
# Path sanitisation
# ---------------------------------------------------------------------------

_SAFE_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_path_component(value: str) -> str:
    """Replace disallowed chars with '_', reject '..' and absolute paths."""
    # Reject '..' or absolute paths BEFORE sanitization
    if value.startswith("/") or value.startswith("\\"):
        raise ArtifactPathError(f"Absolute path component rejected: {value!r}")
    # Reject any '..' component anywhere in the input
    parts = value.replace("\\", "/").split("/")
    if ".." in parts or "." in (p for p in parts if p == "."):
        raise ArtifactPathError(f"Invalid path component (contains '..'): {value!r}")
    if value == "." or value == "..":
        raise ArtifactPathError(f"Invalid path component: {value!r}")
    cleaned = _SAFE_COMPONENT_RE.sub("_", value).strip("_")
    if not cleaned or cleaned in (".", ".."):
        raise ArtifactPathError(f"Invalid path component after sanitise: {value!r}")
    return cleaned


def _validate_under_root(path: Path, root: Path) -> None:
    """Reject paths that escape root via `..`, symlinks, or absolute components."""
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path.resolve()
    resolved_root = root.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise ArtifactPathError(
            f"Path escapes artifact root: {path} resolves to {resolved} (not under {resolved_root})"
        ) from None
    # Check each component for `..` before resolution
    for part in path.parts:
        if part == "..":
            raise ArtifactPathError(f"Path contains '..' component: {path}")
    if path.is_absolute() and not str(path).startswith(str(root)):
        raise ArtifactPathError(f"Absolute path not under artifact root: {path}")


def _evaluation_layer(
    project: str | None,
    dataset: str | None,
    source_run_id: str | None,
) -> Path:
    """Return the evaluation subdirectory under an artifact root.

    Flat legacy layout (``evaluation/``) is used when neither ``project`` nor
    ``source_run_id`` is given; ``dataset`` only affects the predictions file
    name in that case. The three-layer mirror (``evaluation/<project>/<dataset>/
    <source_run_id>/``) is used only when all three are provided — anything in
    between is a configuration error so a directory can always be reversed back
    into its lineage unambiguously.
    """
    if project is None and source_run_id is None:
        return Path("evaluation")
    if project is None or dataset is None or source_run_id is None:
        raise ArtifactPathError(
            "project, dataset, and source_run_id must be provided together "
            "(all-or-nothing) to build a three-layer evaluation path"
        )
    return (
        Path("evaluation")
        / sanitize_path_component(project)
        / sanitize_path_component(dataset)
        / sanitize_path_component(source_run_id)
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    """Return SHA-256 hex digest of canonical JSON serialisation."""
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Layout value objects (pure path computation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionArtifactLayout:
    """Paths for predictions within an artifact run root."""

    artifact_run_root: Path
    evaluation_dir: Path
    predictions_dir: Path
    predictions_file: Path
    conversion_manifest_file: Path


@dataclass(frozen=True, slots=True)
class JudgeRunArtifactLayout:
    """Paths for a single Judge run within an artifact run root."""

    judge_model_dir: Path
    judge_run_dir: Path
    generation_file: Path
    retrieval_file: Path
    indexing_file: Path
    summary_file: Path
    run_manifest_file: Path
    generation_partial: Path
    retrieval_partial: Path
    latest_pointer: Path


# ---------------------------------------------------------------------------
# Layout resolver
# ---------------------------------------------------------------------------


class ArtifactLayoutResolver:
    """Compute standard evaluation/predictions + evaluation/llmjudge paths."""

    @staticmethod
    def predictions(
        artifact_run_root: Path,
        dataset: str,
        predictions_dir: Path | None = None,
        project: str | None = None,
        source_run_id: str | None = None,
    ) -> PredictionArtifactLayout:
        root = artifact_run_root.resolve()
        ds_safe = sanitize_path_component(dataset)
        if predictions_dir is None:
            eval_dir = root / _evaluation_layer(project, dataset, source_run_id)
            pred_dir = eval_dir / "predictions"
        else:
            pred_dir = predictions_dir.resolve()
            eval_dir = pred_dir.parent
            _validate_under_root(pred_dir, root)
        pred_file = pred_dir / f"predictions_{ds_safe}.json"
        manifest_file = pred_dir / "conversion_manifest.json"
        _validate_under_root(pred_file, root)
        _validate_under_root(manifest_file, root)
        return PredictionArtifactLayout(
            artifact_run_root=root,
            evaluation_dir=eval_dir,
            predictions_dir=pred_dir,
            predictions_file=pred_file,
            conversion_manifest_file=manifest_file,
        )

    @staticmethod
    def judge_run(
        artifact_run_root: Path,
        judge_model: str,
        run_id: str,
        project: str | None = None,
        dataset: str | None = None,
        source_run_id: str | None = None,
    ) -> JudgeRunArtifactLayout:
        root = artifact_run_root.resolve()
        model_safe = sanitize_path_component(judge_model)
        run_safe = sanitize_path_component(run_id)
        eval_layer = _evaluation_layer(project, dataset, source_run_id)
        model_dir = root / eval_layer / "llmjudge" / model_safe
        run_dir = model_dir / run_safe
        _validate_under_root(run_dir, root / eval_layer)
        return JudgeRunArtifactLayout(
            judge_model_dir=model_dir,
            judge_run_dir=run_dir,
            generation_file=run_dir / "generation_results.json",
            retrieval_file=run_dir / "retrieval_results.json",
            indexing_file=run_dir / "indexing_results.json",
            summary_file=run_dir / "summary.json",
            run_manifest_file=run_dir / "run_manifest.json",
            generation_partial=run_dir / "generation_results.json.partial",
            retrieval_partial=run_dir / "retrieval_results.json.partial",
            latest_pointer=model_dir / "latest.json",
        )

    @staticmethod
    def find_judge_run(
        artifact_run_root: Path,
        run_id: str,
        project: str | None = None,
        dataset: str | None = None,
        source_run_id: str | None = None,
    ) -> tuple[JudgeRunArtifactLayout, dict[str, Any]] | None:
        """Find an existing run by ID across Judge model directories.

        A resume ID identifies a persisted run, not a newly selected model
        directory. This lookup keeps deterministic-only and LLM-backed
        metric updates in the same run directory.

        With the three-layer mirror (all of ``project``/``dataset``/
        ``source_run_id`` given) the scan is bounded to that single lineage
        directory; otherwise the legacy flat ``evaluation/llmjudge/`` layout is
        scanned.
        """
        root = artifact_run_root.resolve()
        run_safe = sanitize_path_component(run_id)
        eval_layer = _evaluation_layer(project, dataset, source_run_id)
        judge_root = root / eval_layer / "llmjudge"
        if not judge_root.is_dir():
            return None

        candidates: list[tuple[JudgeRunArtifactLayout, dict[str, Any]]] = []
        for model_dir in sorted(judge_root.iterdir()):
            if not model_dir.is_dir():
                continue
            manifest_file = model_dir / run_safe / "run_manifest.json"
            if not manifest_file.is_file():
                continue
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise ArtifactPathError(
                    f"Cannot load existing Judge manifest: {manifest_file}: {exc}"
                ) from exc
            if manifest.get("judge_run_id", run_id) != run_id:
                continue
            layout = ArtifactLayoutResolver.judge_run(
                root,
                model_dir.name,
                run_id,
                project=project,
                dataset=dataset,
                source_run_id=source_run_id,
            )
            candidates.append((layout, manifest))

        if len(candidates) > 1:
            model_dirs = [str(layout.judge_model_dir) for layout, _ in candidates]
            raise ArtifactPathError(
                f"Judge run ID is ambiguous across model directories: {run_id!r}: {model_dirs}"
            )
        return candidates[0] if candidates else None

    @staticmethod
    def infer_run_root(predictions_file: Path) -> Path:
        """Infer artifact_run_root from a standard predictions path.

        Accepts either the legacy flat layout ending with
        ``evaluation/predictions/predictions_*.json`` or the three-layer mirror
        ending with ``evaluation/<project>/<dataset>/<run_id>/predictions/
        predictions_*.json``.
        """
        p = predictions_file.resolve()
        parts = p.parts
        if len(parts) < 3:
            raise ArtifactPathError(f"Cannot infer artifact run root from {predictions_file}")
        if parts[-1].startswith("predictions_") and parts[-2] == "predictions":
            if parts[-3] == "evaluation":
                # legacy flat layout: .../evaluation/predictions/predictions_*.json
                return Path(*parts[:-3])
            if len(parts) >= 7 and parts[-6] == "evaluation":
                # three-layer mirror: .../evaluation/<project>/<dataset>/<run_id>/
                #   predictions/predictions_*.json
                return Path(*parts[:-6])
        raise ArtifactPathError(
            f"Predictions file is not under evaluation/predictions/ or "
            f"evaluation/<project>/<dataset>/<run_id>/predictions/: {predictions_file}"
        )

    @staticmethod
    def infer_lineage(
        predictions_file: Path,
    ) -> tuple[str | None, str | None, str | None]:
        """Infer ``(project, dataset, source_run_id)`` from a predictions path.

        Returns ``(None, None, None)`` for the legacy flat layout, so callers
        can distinguish a mirrored source-run directory from the flat layout.
        """
        p = predictions_file.resolve()
        parts = p.parts
        if (
            len(parts) >= 7
            and parts[-6] == "evaluation"
            and parts[-2] == "predictions"
            and parts[-1].startswith("predictions_")
        ):
            return parts[-5], parts[-4], parts[-3]
        return None, None, None


# ---------------------------------------------------------------------------
# Judge artifact store (I/O, delegates atomic writes to checkpoint)
# ---------------------------------------------------------------------------


class JudgeArtifactStore:
    """Persist predictions, manifests, results, and latest pointers."""

    def write_predictions(
        self, layout: PredictionArtifactLayout, rows: list[dict[str, Any]]
    ) -> None:
        layout.predictions_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(rows, str(layout.predictions_file))

    def write_conversion_manifest(
        self, layout: PredictionArtifactLayout, manifest: dict[str, Any]
    ) -> None:
        layout.predictions_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest, str(layout.conversion_manifest_file))

    def write_result(self, layout: JudgeRunArtifactLayout, kind: str, data: dict[str, Any]) -> None:
        layout.judge_run_dir.mkdir(parents=True, exist_ok=True)
        file_map = {
            "generation": layout.generation_file,
            "retrieval": layout.retrieval_file,
            "indexing": layout.indexing_file,
        }
        target = file_map.get(kind)
        if target is None:
            raise ValueError(f"Unknown result kind: {kind}")
        atomic_write_json(data, str(target))

    def write_summary(self, layout: JudgeRunArtifactLayout, summary: dict[str, Any]) -> None:
        atomic_write_json(summary, str(layout.summary_file))

    def write_run_manifest(self, layout: JudgeRunArtifactLayout, manifest: dict[str, Any]) -> None:
        atomic_write_json(manifest, str(layout.run_manifest_file))

    def update_latest(self, layout: JudgeRunArtifactLayout, run_id: str) -> None:
        layout.judge_model_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "schema_version": 1,
            "latest_run_id": run_id,
            "latest_run_dir": str(layout.judge_run_dir.relative_to(layout.judge_model_dir)),
        }
        atomic_write_json(data, str(layout.latest_pointer))

    def load_for_resume(self, layout: JudgeRunArtifactLayout) -> dict[str, Any] | None:
        """Load existing run manifest for resume validation."""
        if not layout.run_manifest_file.exists():
            return None
        try:
            with open(layout.run_manifest_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
