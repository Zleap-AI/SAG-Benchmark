import asyncio

from pipeline.utils.llm_tracking import LLMTokenTracker
from pipeline.utils.run_stats import RunStatsTracker


def test_run_stats_normalizes_timing_and_token_fields(monkeypatch):
    perf_counter_values = iter([10.0, 12.0, 15.5, 20.0])
    monkeypatch.setattr(
        "pipeline.utils.run_stats.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    tracker = RunStatsTracker()
    with tracker.measure("extract") as measurement:
        pass

    asyncio.run(
        tracker.token_tracker.record(
            "extract",
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
        )
    )

    summary = tracker.get_summary()

    assert measurement.elapsed_seconds == 3.5
    assert summary["timing"] == {
        "total_seconds": 10.0,
        "stages": {"extract": {"seconds": 3.5}},
    }
    assert summary["tokens"] == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "calls": 1,
        "stages": {
            "extract": {
                "calls": 1,
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            }
        },
    }


def test_token_snapshot_delta_is_per_file_not_cumulative():
    tracker = RunStatsTracker()
    before = tracker.token_snapshot()

    asyncio.run(
        tracker.token_tracker.record(
            "extract",
            {"input_tokens": 80, "output_tokens": 20},
        )
    )
    after = tracker.token_snapshot()

    assert tracker.token_delta(before, after) == {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "calls": 1,
    }


def test_legacy_token_summary_remains_available():
    tracker = LLMTokenTracker()
    asyncio.run(
        tracker.record(
            "EXTRACT",
            {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
    )

    assert tracker.get_summary() == {
        "total_prompt": 7,
        "total_completion": 3,
        "total_tokens": 10,
        "total_calls": 1,
        "stages": {
            "EXTRACT": {
                "calls": 1,
                "prompt": 7,
                "completion": 3,
                "total": 10,
            }
        },
    }
