import asyncio

import pytest

from pipeline.modules.search.sag2 import SAG2_EVENT_STATS_KEYS, SAG2_TIMING_STAGE_ORDER
from pipeline.evaluation.utils.step_timing import compute_step_timings
from scripts.run_search_benchmark import _emit_progress, _log_and_report_step_timings


class _Logger:
    def info(self, _message):
        return None


class _CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _Tracker:
    def __init__(self):
        self.calls = []
        self.supplementary = None

    def log_evaluation_metrics(self, *_args, **_kwargs):
        self.calls.append("evaluation")

    def log_recall_metrics(self, *_args, **_kwargs):
        self.calls.append("recall")

    def log_supplementary_metrics(self, metrics, _step):
        self.calls.append("supplementary")
        self.supplementary = metrics


def _timed_result() -> dict:
    with_retry = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
    no_retry = dict(with_retry)
    with_retry["rewrite_query"] = 2.0
    no_retry["rewrite_query"] = 1.0
    return {
        "sections": ["doc"],
        "timing_steps": {
            "schema_version": 2,
            "stages_with_retry": with_retry,
            "stages_no_retry": no_retry,
            "wall_total_observed": 2.0,
            "step7_llm_calls": 1,
            "step7_prompt_tokens": 10,
            "step7_completion_tokens": 5,
            **with_retry,
            "total": 2.0,
        },
        "event_stats": dict.fromkeys(SAG2_EVENT_STATS_KEYS, 1),
    }


@pytest.mark.asyncio
async def test_progress_logs_core_metrics_before_ordered_supplementary_metrics():
    tracker = _Tracker()

    await _emit_progress(
        current_idx=1,
        total=1,
        ordered_results=[_timed_result()],
        gold_docs_for_recall=[["doc"]],
        bench_size=1,
        mlflow_tracker=tracker,
        bench_logger=_Logger(),
        prev_stats={},
        stats_lock=asyncio.Lock(),
    )

    assert tracker.calls == ["evaluation", "recall", "supplementary"]
    assert tracker.supplementary is not None
    prefixes = []
    for key in tracker.supplementary:
        prefix = "_".join(key.split("_", 2)[:2])
        if not prefixes or prefixes[-1] != prefix:
            prefixes.append(prefix)
    assert prefixes == [
        "supp_01",
        "supp_02",
        "supp_03",
        "supp_04",
    ]
    assert not any("_time_batch_" in key or "percall" in key for key in tracker.supplementary)
    assert "supp_03_token_cum_total_sum" in tracker.supplementary
    assert tracker.supplementary["supp_04_token_batch_total_sum"] == 15.0
    assert not any(
        key.startswith(("supp_03_event", "supp_04_event"))
        for key in tracker.supplementary
    )


def test_schema_v2_timing_diagnostics_are_not_printed_to_console():
    logger = _CapturingLogger()
    tracker = _Tracker()
    cumulative = compute_step_timings([_timed_result()])

    _log_and_report_step_timings(
        logger,
        cumulative,
        batch_view=None,
        mlflow_tracker=tracker,
        batch_index=1,
    )

    assert logger.messages == []
    assert tracker.calls == ["supplementary"]
    assert tracker.supplementary is not None
