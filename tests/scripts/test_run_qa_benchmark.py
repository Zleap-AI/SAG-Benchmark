import logging
from pathlib import Path

from scripts.run_qa_benchmark import (
    _QA_FILE_HANDLER_MARKER,
    _install_qa_file_handler,
    _qa_results_path,
    _resolve_qa_output_dir,
)


def _remove_handler(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def test_default_output_keeps_run_log_and_results_in_timestamped_qa_directory(tmp_path: Path):
    search_run = tmp_path / "search_run"
    search_run.mkdir()
    input_path = search_run / "search_results.json"
    parent_log = search_run / "run.log"
    parent_log.write_text("search benchmark sentinel\n", encoding="utf-8")
    output_dir = _resolve_qa_output_dir(input_path, None, "20260806_152023")
    output_dir.mkdir(parents=True)
    root = logging.getLogger()
    original_level = root.level
    non_qa_handlers = [
        handler for handler in root.handlers if not getattr(handler, _QA_FILE_HANDLER_MARKER, False)
    ]

    handler, log_file = _install_qa_file_handler(output_dir)
    try:
        root.warning("qa-default-directory-marker")
        handler.flush()
        assert output_dir == search_run / "qa_20260806_152023"
        assert log_file == output_dir / "run.log"
        assert _qa_results_path(output_dir) == output_dir / "qa_results.json"
        assert "qa-default-directory-marker" in log_file.read_text(encoding="utf-8")
        assert parent_log.read_text(encoding="utf-8") == "search benchmark sentinel\n"
        assert root.level == original_level
        assert all(existing in root.handlers for existing in non_qa_handlers)
    finally:
        _remove_handler(handler)


def test_explicit_output_directory_is_used_for_both_artifacts(tmp_path: Path):
    input_path = tmp_path / "search" / "search_results.json"
    output_dir = _resolve_qa_output_dir(input_path, str(tmp_path / "custom"), "ignored")
    output_dir.mkdir(parents=True)

    handler, log_file = _install_qa_file_handler(output_dir)
    try:
        assert output_dir == tmp_path / "custom"
        assert log_file.parent == output_dir
        assert _qa_results_path(output_dir).parent == output_dir
    finally:
        _remove_handler(handler)


def test_reinitialization_closes_old_handler_and_stops_cross_directory_writes(tmp_path: Path):
    first_dir = tmp_path / "qa_first"
    second_dir = tmp_path / "qa_second"
    first_dir.mkdir()
    second_dir.mkdir()
    root = logging.getLogger()

    first_handler, first_log = _install_qa_file_handler(first_dir)
    root.warning("first-only-marker")
    first_handler.flush()
    second_handler, second_log = _install_qa_file_handler(second_dir)
    try:
        assert first_handler.stream is None
        root.warning("second-only-marker")
        second_handler.flush()
        assert "first-only-marker" in first_log.read_text(encoding="utf-8")
        assert "second-only-marker" not in first_log.read_text(encoding="utf-8")
        assert "second-only-marker" in second_log.read_text(encoding="utf-8")
    finally:
        _remove_handler(second_handler)
