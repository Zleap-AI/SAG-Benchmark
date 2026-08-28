"""Shared Step 0 dataset normalization and artifact export.

Raw QA/corpus files are read only through :class:`DatasetLoader`. External
projects keep their own caches because those files are method runtime inputs,
not independent copies of the source dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .load_utils import DatasetLoader
except ImportError:  # direct import from external projects' isolated environments
    from load_utils import DatasetLoader


@dataclass(frozen=True)
class ReproduceDatasetPaths:
    cache_root: Path
    dataset_name: str
    subdir: str | None = None  # 目录层覆盖（如 upload 的 source_config_id）；None 时用 dataset_name

    @property
    def dataset_dir(self) -> Path:
        return self.cache_root / (self.subdir or self.dataset_name)

    @property
    def contexts_dir(self) -> Path:
        return self.dataset_dir / "contexts"

    @property
    def questions_dir(self) -> Path:
        return self.dataset_dir / "questions"

    @property
    def docs_path(self) -> Path:
        return self.contexts_dir / f"{self.dataset_name}_corpus_docs.json"

    @property
    def questions_path(self) -> Path:
        return self.questions_dir / f"{self.dataset_name}_questions.json"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "dataset_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class ReproduceDatasetExporter:
    """Export one canonical dataset pair into a method-local cache layout."""

    def __init__(self, loader: DatasetLoader, cache_root: str | Path, subdir: str | None = None):
        self.loader = loader
        self.paths = ReproduceDatasetPaths(
            Path(cache_root).resolve(), loader.dataset_name, subdir=subdir
        )

    def export(
        self,
        *,
        legacy_sidecars: bool = False,
        limit_documents: int = 0,
        limit_questions: int = 0,
    ) -> dict[str, Any]:
        samples_path, corpus_path = self.loader.validate_source_pair()
        docs = self.loader.get_docs()
        questions = self.loader.get_question_records()
        if limit_documents:
            docs = docs[:limit_documents]
        if limit_questions:
            questions = questions[:limit_questions]

        atomic_write_json(self.paths.docs_path, docs)
        atomic_write_json(self.paths.questions_path, questions)

        legacy_paths: list[str] = []
        if legacy_sidecars:
            question_path = self.paths.questions_dir / f"{self.loader.dataset_name}_stage.json"
            reference_path = self.paths.questions_dir / f"{self.loader.dataset_name}_stage_ref.json"
            atomic_write_json(question_path, [item["question"] for item in questions])
            atomic_write_json(reference_path, [item["gold_ref"] for item in questions])
            legacy_paths = [str(question_path), str(reference_path)]

        manifest = {
            "schema_version": 1,
            "dataset_name": self.loader.dataset_name,
            "source": {
                "dataset_root": str(self.loader.dataset_dir.resolve()),
                "samples": str(samples_path.resolve()),
                "samples_sha256": _sha256(samples_path),
                "corpus": str(corpus_path.resolve()),
                "corpus_sha256": _sha256(corpus_path),
            },
            "counts": {"documents": len(docs), "questions": len(questions)},
            "limits": {
                "documents": limit_documents or None,
                "questions": limit_questions or None,
            },
            "artifacts": {
                "documents": str(self.paths.docs_path),
                "questions": str(self.paths.questions_path),
                "legacy_sidecars": legacy_paths,
            },
        }
        atomic_write_json(self.paths.manifest_path, manifest)
        return manifest


def export_reproduce_dataset(
    dataset_name: str,
    cache_root: str | Path,
    *,
    dataset_root: str | Path | None = None,
    legacy_sidecars: bool = False,
    limit_documents: int = 0,
    limit_questions: int = 0,
    subdir: str | None = None,
) -> dict[str, Any]:
    """Convenience entry point used by thin external Step 0 wrappers.

    subdir: 覆盖产物目录层（如 upload 的 source_config_id）；None 时用 dataset_name。
    """
    loader = DatasetLoader(dataset_name, dataset_root)
    return ReproduceDatasetExporter(loader, cache_root, subdir=subdir).export(
        legacy_sidecars=legacy_sidecars,
        limit_documents=limit_documents,
        limit_questions=limit_questions,
    )
