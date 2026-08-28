import os
from pathlib import Path

import pytest

from scripts.run_benchmark import find_latest_results


def _result(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_find_latest_results_uses_canonical_output_layout(tmp_path: Path) -> None:
    older = _result(tmp_path / "musique" / "vector" / "run-a" / "search_results.json", 1)
    latest = _result(tmp_path / "musique" / "sag2" / "run-b" / "search_results.json", 2)
    _result(tmp_path / "hotpotqa" / "sag2" / "run-c" / "search_results.json", 3)

    assert find_latest_results("musique", tmp_path) == latest
    assert older != latest


def test_find_latest_results_rejects_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="musique"):
        find_latest_results("musique", tmp_path)
