import pytest

from pipeline.evaluation.utils.step_timing import (
    _build_timing_mlflow_metrics,
    batch_view_from_cumulative,
    build_supplementary_metrics,
    compute_step_timings,
    raw_diagnostic_snapshot,
)
from pipeline.modules.search.sag2 import SAG2_EVENT_STATS_KEYS, SAG2_TIMING_STAGE_ORDER


def _result(
    rewrite: float,
    rewrite_no_retry: float,
    finalize: float,
    *,
    prompt: int,
    completion: int,
    calls: int,
    event_multiplier: int,
) -> dict:
    with_retry = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
    no_retry = dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0)
    with_retry.update({"rewrite_query": rewrite, "finalize": finalize})
    no_retry.update({"rewrite_query": rewrite_no_retry, "finalize": finalize})
    return {
        "timing_steps": {
            "schema_version": 2,
            "stage_order": list(SAG2_TIMING_STAGE_ORDER),
            "stages_with_retry": with_retry,
            "stages_no_retry": no_retry,
            "retry_wasted_by_stage": {
                stage: with_retry[stage] - no_retry[stage]
                for stage in SAG2_TIMING_STAGE_ORDER
            },
            "total_with_retry": sum(with_retry.values()),
            "total_no_retry": sum(no_retry.values()),
            "retry_wasted_total": rewrite - rewrite_no_retry,
            "wall_total_observed": sum(with_retry.values()),
            "step7_llm_calls": calls,
            "step7_prompt_tokens": prompt,
            "step7_completion_tokens": completion,
            # Legacy compatibility fields are deliberately present.
            **with_retry,
            "total": sum(with_retry.values()),
        },
        "event_stats": {
            key: (index + 1) * event_multiplier
            for index, key in enumerate(SAG2_EVENT_STATS_KEYS)
        },
    }


def _assert_time_conservation(view: dict) -> None:
    with_retry = view["time"]["with_retry"]
    no_retry = view["time"]["no_retry"]
    assert sum(with_retry["stages"].values()) == pytest.approx(with_retry["total"])
    assert sum(no_retry["stages"].values()) == pytest.approx(no_retry["total"])
    assert with_retry["total"] - no_retry["total"] == pytest.approx(
        view["time"]["retry_wasted"]["total"]
    )


def test_schema_v2_cumulative_uses_one_completed_denominator_and_conserves_totals():
    result1 = _result(1.0, 0.5, 1.0, prompt=10, completion=2, calls=1, event_multiplier=1)
    result2 = _result(3.0, 3.0, 1.0, prompt=20, completion=4, calls=2, event_multiplier=2)

    view = compute_step_timings([result1, result2])

    assert view["schema_version"] == 2
    assert view["conservation_guaranteed"] is True
    assert view["completed"] == 2
    assert view["time"]["with_retry"]["stages"]["rewrite_query"] == 2.0
    assert view["time"]["with_retry"]["stages"]["candidate_pool"] == 0.0
    assert view["time"]["with_retry"]["total"] == 3.0
    assert view["time"]["no_retry"]["total"] == 2.75
    _assert_time_conservation(view)

    # Event counters remain attached to each search result and are not
    # aggregated into batch/cumulative timing diagnostics.
    assert "event" not in view
    assert view["token"]["sum"] == {"prompt": 30.0, "completion": 6.0, "total": 36.0}
    assert view["token"]["avg_per_question"]["total"] == 18.0
    assert view["token"]["avg_per_call"]["total"] == 12.0
    assert view["token"]["calls"] == 3


def test_batch_is_cumulative_raw_snapshot_difference_not_question_index_slice():
    first = _result(1.0, 0.5, 1.0, prompt=10, completion=2, calls=1, event_multiplier=1)
    later_completed_lower_index = _result(
        3.0, 3.0, 1.0, prompt=20, completion=4, calls=2, event_multiplier=2
    )
    previous_cumulative = compute_step_timings([None, first])
    previous_snapshot = raw_diagnostic_snapshot(previous_cumulative)
    current_cumulative = compute_step_timings([later_completed_lower_index, first])

    batch = batch_view_from_cumulative(current_cumulative, previous_snapshot)

    assert batch is not None
    assert batch["completed"] == 1
    assert batch["time"]["with_retry"]["stages"]["rewrite_query"] == 3.0
    assert batch["time"]["no_retry"]["stages"]["rewrite_query"] == 3.0
    assert batch["token"]["sum"]["total"] == 24.0
    assert "event" not in batch
    _assert_time_conservation(batch)


def test_supplementary_metrics_export_only_cumulative_additive_time_groups():
    first = _result(1.0, 0.5, 1.0, prompt=10, completion=2, calls=1, event_multiplier=1)
    second = _result(3.0, 3.0, 1.0, prompt=20, completion=4, calls=2, event_multiplier=2)
    cum = compute_step_timings([first, second])
    batch = batch_view_from_cumulative(cum, raw_diagnostic_snapshot(compute_step_timings([first])))

    metrics = build_supplementary_metrics(cum, batch)
    prefixes = []
    for key in metrics:
        prefix = key.split("_", 2)[:2]
        normalized = "_".join(prefix)
        if not prefixes or prefixes[-1] != normalized:
            prefixes.append(normalized)
    assert prefixes == [
        "supp_01",
        "supp_02",
        "supp_03",
        "supp_04",
    ]
    for prefix in (
        "supp_01_time_cum_with_retry",
        "supp_02_time_cum_no_retry",
    ):
        stages = [metrics[f"{prefix}_{index:02d}_{stage}"] for index, stage in enumerate(
            SAG2_TIMING_STAGE_ORDER, start=1
        )]
        assert sum(stages) == pytest.approx(metrics[f"{prefix}_99_total"])

    assert not any("_time_batch_" in key or "percall" in key for key in metrics)
    assert not any(key.startswith(("supp_03_event", "supp_04_event")) for key in metrics)
    assert "supp_03_token_cum_total_sum" in metrics
    assert "supp_04_token_batch_total_sum" in metrics


def test_legacy_timing_payload_remains_available_without_conservation_claim():
    view = compute_step_timings(
        [
            {
                "timing_steps": {
                    "step1_extract": 1.0,
                    "step7_llm_no_retry": 2.0,
                    "step7_llm_with_retry": 3.0,
                    "step7_llm_calls": 1,
                    "step7_prompt_tokens": 10,
                    "step7_completion_tokens": 5,
                    "total": 4.0,
                }
            }
        ]
    )

    assert view["schema_version"] == 1
    assert view["conservation_guaranteed"] is False
    assert view["completed"] == 1
    assert view["per_question"]["step1_extract"] == 1.0
    assert view["step7_per_call"]["with_retry"] == 3.0
    mlflow_metrics = _build_timing_mlflow_metrics(view, suffix="_cum")
    assert mlflow_metrics["step1_extract_cum"] == 1.0
    assert not any("percall" in key or "batch" in key for key in mlflow_metrics)


def test_legacy_consecutive_batches_use_incremental_completed_count():
    first = {
        "timing_steps": {
            "step1_extract": 1.0,
            "step7_llm_calls": 1,
            "total": 1.0,
        }
    }
    second = {
        "timing_steps": {
            "step1_extract": 3.0,
            "step7_llm_calls": 2,
            "total": 3.0,
        }
    }
    first_cumulative = compute_step_timings([first])
    snapshot = raw_diagnostic_snapshot(first_cumulative)
    second_cumulative = compute_step_timings([first, second])

    batch = batch_view_from_cumulative(second_cumulative, snapshot)

    assert batch is not None
    assert batch["schema_version"] == 1
    assert batch["completed"] == 1
    assert batch["per_question"]["step1_extract"] == 3.0
    assert batch["step7_per_call"]["calls"] == 2
