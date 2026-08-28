"""Tests for checkpoint: atomic write, NaN detection, resume, partial merge."""

import json
import os
import tempfile

import pytest

from pipeline.evaluation.judge.checkpoint import (
    _entry_has_nan,
    _select_rerun_ids,
    atomic_write_json,
    load_checkpoint,
    merge_partial_results,
)


class TestAtomicWrite:
    def test_creates_missing_parent_directory(self, tmp_path):
        path = tmp_path / "new" / "judge" / "checkpoint.json"

        atomic_write_json({"complete": 5}, str(path))

        assert json.loads(path.read_text(encoding="utf-8")) == {"complete": 5}

    def test_write_and_read(self):
        data = {"key": "value", "nested": {"a": 1}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            atomic_write_json(data, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == data
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_overwrite_existing(self):
        data1 = {"version": 1}
        data2 = {"version": 2}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            atomic_write_json(data1, path)
            atomic_write_json(data2, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == data2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_original_preserved_on_failure(self):
        data1 = {"original": True}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            atomic_write_json(data1, path)
            # Try writing to an invalid path to verify original is untouched
            with pytest.raises(OSError):
                atomic_write_json({"bad": True}, "/nonexistent/dir/file.json")
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == data1
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestEntryHasNaN:
    def test_no_nan(self):
        entry = {"id": 1, "metrics": {"qa_em": 1.0, "qa_f1": 0.8}}
        assert not _entry_has_nan(entry)

    def test_with_nan(self):
        entry = {"id": 1, "metrics": {"qa_em": float("nan"), "qa_f1": 0.8}}
        assert _entry_has_nan(entry)

    def test_empty_metrics(self):
        entry = {"id": 1, "metrics": {}}
        assert not _entry_has_nan(entry)

    def test_no_metrics_key(self):
        entry = {"id": 1}
        assert not _entry_has_nan(entry)


class TestSelectRerunIds:
    def test_force_all(self):
        old = [{"id": 0, "metrics": {"qa_em": 1.0}}]
        ids = _select_rerun_ids(old, {0, 1, 2}, force_all=True)
        assert ids == {0, 1, 2}

    def test_nan_only(self):
        old = [
            {"id": 0, "metrics": {"qa_em": float("nan")}},
            {"id": 1, "metrics": {"qa_em": 1.0}},
        ]
        ids = _select_rerun_ids(old, {0, 1, 2}, force_all=False)
        assert ids == {0, 2}  # id=0 has NaN, id=2 is missing

    def test_no_nan_no_missing(self):
        old = [
            {"id": 0, "metrics": {"qa_em": 1.0}},
            {"id": 1, "metrics": {"qa_em": 0.8}},
        ]
        ids = _select_rerun_ids(old, {0, 1}, force_all=False)
        assert ids == set()


class TestLoadCheckpoint:
    def test_nonexistent(self):
        result = load_checkpoint("/tmp/nonexistent_ckpt_test.json")
        assert result is None

    def test_valid(self):
        data = {"key": "value"}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_checkpoint(path)
            assert result == data
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_corrupted(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json {{{")
            path = f.name
        try:
            result = load_checkpoint(path)
            assert result is None
            # Corrupted file should still exist (not deleted)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestMergePartialResults:
    def test_full_replace(self):
        existing = {
            "qa": {
                "average_scores": {"qa_em": 0.5, "qa_f1": 0.6},
                "detailed": [
                    {"id": 0, "metrics": {"qa_em": 0.5, "qa_f1": 0.6}},
                    {"id": 1, "metrics": {"qa_em": 0.5, "qa_f1": 0.6}},
                ],
            }
        }
        new_results = {
            "detailed": [
                {"id": 0, "metrics": {"qa_em": 1.0, "qa_f1": 1.0}},
            ]
        }
        merged = merge_partial_results(existing, {"qa": new_results}, "qa", {0}, None)
        # id=0 should be updated, id=1 preserved
        detailed = merged["detailed"]
        by_id = {d["id"]: d for d in detailed}
        assert by_id[0]["metrics"]["qa_em"] == 1.0
        assert by_id[1]["metrics"]["qa_em"] == 0.5
        assert merged["average_scores"]["qa_em"] == 0.75  # (1.0 + 0.5) / 2

    def test_only_metrics_partial_update(self):
        existing = {
            "qa": {
                "average_scores": {"qa_em": 0.5, "qa_f1": 0.6},
                "detailed": [
                    {"id": 0, "metrics": {"qa_em": 0.5, "qa_f1": 0.6}},
                ],
            }
        }
        new_results = {
            "detailed": [
                {"id": 0, "metrics": {"qa_em": 1.0, "rouge_score": 0.9}},
            ]
        }
        merged = merge_partial_results(existing, {"qa": new_results}, "qa", {0}, ["qa_em"])
        detailed = merged["detailed"]
        by_id = {d["id"]: d for d in detailed}
        # qa_em updated, qa_f1 preserved, rouge_score (not in only_metrics) not added
        assert by_id[0]["metrics"]["qa_em"] == 1.0
        assert by_id[0]["metrics"]["qa_f1"] == 0.6
        assert "rouge_score" not in by_id[0]["metrics"]

    def test_new_entry_when_not_in_old(self):
        existing = {
            "qa": {
                "average_scores": {"qa_em": 0.5},
                "detailed": [
                    {"id": 0, "metrics": {"qa_em": 0.5}},
                ],
            }
        }
        new_results = {
            "detailed": [
                {"id": 1, "metrics": {"qa_em": 1.0}},
            ]
        }
        merged = merge_partial_results(existing, {"qa": new_results}, "qa", {1}, None)
        assert len(merged["detailed"]) == 2
