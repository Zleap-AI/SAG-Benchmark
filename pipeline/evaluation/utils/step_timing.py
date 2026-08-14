"""Pure aggregation and presentation helpers for search diagnostics.

SAG2 timing schema v2 uses one fixed additive stage vector, one denominator per
window, and raw cumulative sums that can be differenced safely for concurrent
batch progress.  The legacy multi/hopllm payload and helper APIs remain
available as a non-conservation-guaranteed fallback.
"""

from typing import Any

from pipeline.modules.search.sag2 import SAG2_TIMING_STAGE_ORDER

# Legacy multi/hopllm timing keys.  Keep these names and semantics for callers
# that have not migrated to schema v2.
_STEP_DISPLAY_KEYS = [
    "step0_scope",
    "step1_extract",
    "step2_retrieve_entities",
    "step3_retrieve_events",
    "step4_fetch_details",
    "step5_expand",
    "step6_coarse_rank",
    "step8_chunks",
    "rewrite_query",
    "parallel_recall",
    "event_similarity_filter",
    "expand",
    "score_sort",
    "rerank",
    "answer_graph",
]
_ACCUM_KEYS = _STEP_DISPLAY_KEYS + ["step7_llm_no_retry", "step7_llm_with_retry", "total"]
_TOTAL_KEYS = _STEP_DISPLAY_KEYS + ["step7_llm_no_retry"]
_TOKEN_KEYS = ["step7_prompt_tokens", "step7_completion_tokens"]
_ALL_ACCUM_KEYS = _ACCUM_KEYS + _TOKEN_KEYS


def _avg_from_sums(
    sums: dict[str, float], counts: dict[str, int], keys: list[str], ndigits: int = 4
) -> dict[str, float]:
    """Legacy per-key averages; retained for old payload compatibility."""

    return {k: round(sums[k] / counts[k], ndigits) for k in keys if counts.get(k, 0) > 0}


def _percall(sums: dict[str, float], key: str, calls: int, ndigits: int = 4) -> float:
    return round(sums.get(key, 0.0) / calls, ndigits) if calls > 0 else 0.0


def views_from_sums(
    sums: dict[str, float], counts: dict[str, int], step7_calls: int
) -> dict[str, Any]:
    """Build the legacy timing/token view from old raw sums and counts."""

    per_question = _avg_from_sums(sums, counts, _ACCUM_KEYS)
    step7_per_call = {
        "with_retry": _percall(sums, "step7_llm_with_retry", step7_calls),
        "no_retry": _percall(sums, "step7_llm_no_retry", step7_calls),
        "calls": step7_calls,
    }
    total_step7_no_retry = round(sum(per_question.get(k, 0.0) for k in _TOTAL_KEYS), 4)
    token_per_question = _avg_from_sums(sums, counts, _TOKEN_KEYS, ndigits=2)
    token_per_call = {
        "prompt": _percall(sums, "step7_prompt_tokens", step7_calls, ndigits=2),
        "completion": _percall(sums, "step7_completion_tokens", step7_calls, ndigits=2),
        "calls": step7_calls,
    }
    return {
        "per_question": per_question,
        "step7_per_call": step7_per_call,
        "total_step7_no_retry": total_step7_no_retry,
        "token_per_question": token_per_question,
        "token_per_call": token_per_call,
    }


def _empty_timing_sums() -> dict[str, Any]:
    return {
        "with_retry": dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0),
        "no_retry": dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0),
        "retry_wasted_by_stage": dict.fromkeys(SAG2_TIMING_STAGE_ORDER, 0.0),
        "total_with_retry": 0.0,
        "total_no_retry": 0.0,
        "retry_wasted_total": 0.0,
        "wall_total_observed": 0.0,
    }


def _empty_token_sums() -> dict[str, float]:
    return {"prompt": 0.0, "completion": 0.0, "total": 0.0}


def _safe_average(value: float, denominator: int) -> float:
    return float(value) / denominator if denominator > 0 else 0.0


def diagnostic_view_from_raw(
    timing_sums: dict[str, Any],
    token_sums: dict[str, float],
    completed: int,
    step7_calls: int,
    *,
    conservation_guaranteed: bool = True,
) -> dict[str, Any]:
    """Build one cumulative or batch schema-v2 view from raw additive sums.

    Every timing stage uses ``completed`` as the denominator.  Totals are
    derived from the resulting stage averages so each displayed/MLflow group
    remains additive, including windows containing skipped stages.
    """

    with_retry = {
        stage: _safe_average(timing_sums["with_retry"].get(stage, 0.0), completed)
        for stage in SAG2_TIMING_STAGE_ORDER
    }
    no_retry = {
        stage: _safe_average(timing_sums["no_retry"].get(stage, 0.0), completed)
        for stage in SAG2_TIMING_STAGE_ORDER
    }
    retry_wasted = {
        stage: with_retry[stage] - no_retry[stage] for stage in SAG2_TIMING_STAGE_ORDER
    }
    total_with_retry = sum(with_retry.values())
    total_no_retry = sum(no_retry.values())

    prompt_sum = float(token_sums.get("prompt", 0.0))
    completion_sum = float(token_sums.get("completion", 0.0))
    total_token_sum = prompt_sum + completion_sum
    token_sum_view = {
        "prompt": prompt_sum,
        "completion": completion_sum,
        "total": total_token_sum,
    }

    return {
        "schema_version": 2,
        "conservation_guaranteed": conservation_guaranteed,
        "completed": completed,
        "time": {
            "with_retry": {"stages": with_retry, "total": total_with_retry},
            "no_retry": {"stages": no_retry, "total": total_no_retry},
            "retry_wasted": {
                "by_stage": retry_wasted,
                "total": total_with_retry - total_no_retry,
            },
            "wall_total_observed": _safe_average(
                timing_sums.get("wall_total_observed", 0.0), completed
            ),
        },
        "token": {
            "sum": token_sum_view,
            "avg_per_question": {
                key: _safe_average(value, completed) for key, value in token_sum_view.items()
            },
            "avg_per_call": {
                key: _safe_average(value, step7_calls) for key, value in token_sum_view.items()
            },
            "calls": step7_calls,
            "completed": completed,
        },
    }


def compute_step_timings(ordered_results: list[dict | None]) -> dict[str, Any]:
    """Aggregate cumulative timing and token diagnostics.

    Schema-v2 questions use fixed keys and the shared ``completed`` denominator.
    Old timing payloads retain the previous per-key fallback view and are marked
    ``conservation_guaranteed=False``.
    """

    timing_sums = _empty_timing_sums()
    token_sums = _empty_token_sums()
    v2_completed = 0
    v2_calls = 0

    legacy_sums: dict[str, float] = {}
    legacy_counts: dict[str, int] = {}
    legacy_calls = 0
    legacy_completed = 0

    for result in ordered_results:
        if result is None:
            continue
        ts = result.get("timing_steps") or {}
        if not ts:
            continue

        for key in _ALL_ACCUM_KEYS:
            value = ts.get(key)
            if isinstance(value, (int, float)):
                legacy_sums[key] = legacy_sums.get(key, 0.0) + float(value)
                legacy_counts[key] = legacy_counts.get(key, 0) + 1
        calls_value = ts.get("step7_llm_calls")
        if isinstance(calls_value, (int, float)):
            legacy_calls += int(calls_value)
        legacy_completed += 1

        if ts.get("schema_version") != 2:
            continue
        stages_with_retry = ts.get("stages_with_retry")
        stages_no_retry = ts.get("stages_no_retry")
        if not isinstance(stages_with_retry, dict) or not isinstance(stages_no_retry, dict):
            continue

        v2_completed += 1
        for stage in SAG2_TIMING_STAGE_ORDER:
            with_value = stages_with_retry.get(stage, 0.0)
            no_value = stages_no_retry.get(stage, 0.0)
            if isinstance(with_value, (int, float)):
                timing_sums["with_retry"][stage] += float(with_value)
            if isinstance(no_value, (int, float)):
                timing_sums["no_retry"][stage] += float(no_value)
            timing_sums["retry_wasted_by_stage"][stage] = (
                timing_sums["with_retry"][stage] - timing_sums["no_retry"][stage]
            )
        timing_sums["total_with_retry"] = sum(timing_sums["with_retry"].values())
        timing_sums["total_no_retry"] = sum(timing_sums["no_retry"].values())
        timing_sums["retry_wasted_total"] = (
            timing_sums["total_with_retry"] - timing_sums["total_no_retry"]
        )
        wall = ts.get("wall_total_observed", 0.0)
        if isinstance(wall, (int, float)):
            timing_sums["wall_total_observed"] += float(wall)

        prompt = ts.get("step7_prompt_tokens", 0)
        completion = ts.get("step7_completion_tokens", 0)
        if isinstance(prompt, (int, float)):
            token_sums["prompt"] += float(prompt)
        if isinstance(completion, (int, float)):
            token_sums["completion"] += float(completion)
        token_sums["total"] = token_sums["prompt"] + token_sums["completion"]
        if isinstance(calls_value, (int, float)):
            v2_calls += int(calls_value)

    if v2_completed:
        view = diagnostic_view_from_raw(
            timing_sums,
            token_sums,
            v2_completed,
            v2_calls,
        )
    else:
        view = views_from_sums(legacy_sums, legacy_counts, legacy_calls)
        view.update(
            {
                "schema_version": 1,
                "conservation_guaranteed": False,
                "completed": legacy_completed,
            }
        )

    # Raw fields are deliberately cumulative.  Progress code snapshots and
    # differences these values instead of slicing by question index.
    view.update(
        {
            "raw_timing_sums": timing_sums,
            "raw_token_sums": token_sums,
            "raw_completed": v2_completed,
            "raw_sums": legacy_sums,
            "raw_counts": legacy_counts,
            "raw_step7_calls": v2_calls if v2_completed else legacy_calls,
        }
    )
    # Preserve legacy presentation fields even for schema v2 consumers.
    if v2_completed:
        view.update(views_from_sums(legacy_sums, legacy_counts, legacy_calls))
        view["completed"] = v2_completed
    return view


def raw_diagnostic_snapshot(view: dict[str, Any]) -> dict[str, Any]:
    """Copy only cumulative raw counters needed for the next batch delta."""

    timing = view.get("raw_timing_sums", _empty_timing_sums())
    schema_version = view.get("schema_version", 1)
    return {
        "schema_version": schema_version,
        "timing_sums": {
            "with_retry": dict(timing.get("with_retry", {})),
            "no_retry": dict(timing.get("no_retry", {})),
            "retry_wasted_by_stage": dict(timing.get("retry_wasted_by_stage", {})),
            "total_with_retry": float(timing.get("total_with_retry", 0.0)),
            "total_no_retry": float(timing.get("total_no_retry", 0.0)),
            "retry_wasted_total": float(timing.get("retry_wasted_total", 0.0)),
            "wall_total_observed": float(timing.get("wall_total_observed", 0.0)),
        },
        "token_sums": dict(view.get("raw_token_sums", {})),
        "completed": int(
            view.get("raw_completed", 0)
            if schema_version == 2
            else view.get("completed", 0)
        ),
        "step7_calls": int(view.get("raw_step7_calls", 0)),
        "legacy_sums": dict(view.get("raw_sums", {})),
        "legacy_counts": dict(view.get("raw_counts", {})),
    }


def _subtract_mapping(current: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    return {key: float(value) - float(previous.get(key, 0.0)) for key, value in current.items()}


def batch_view_from_cumulative(
    cumulative: dict[str, Any], previous_snapshot: dict[str, Any]
) -> dict[str, Any] | None:
    """Build a batch view by differencing two cumulative raw snapshots."""

    current = raw_diagnostic_snapshot(cumulative)
    if cumulative.get("schema_version") == 2:
        previous_timing = previous_snapshot.get("timing_sums", _empty_timing_sums())
        batch_timing = {
            "with_retry": _subtract_mapping(
                current["timing_sums"]["with_retry"], previous_timing.get("with_retry", {})
            ),
            "no_retry": _subtract_mapping(
                current["timing_sums"]["no_retry"], previous_timing.get("no_retry", {})
            ),
            "retry_wasted_by_stage": _subtract_mapping(
                current["timing_sums"]["retry_wasted_by_stage"],
                previous_timing.get("retry_wasted_by_stage", {}),
            ),
            "total_with_retry": current["timing_sums"]["total_with_retry"]
            - float(previous_timing.get("total_with_retry", 0.0)),
            "total_no_retry": current["timing_sums"]["total_no_retry"]
            - float(previous_timing.get("total_no_retry", 0.0)),
            "retry_wasted_total": current["timing_sums"]["retry_wasted_total"]
            - float(previous_timing.get("retry_wasted_total", 0.0)),
            "wall_total_observed": current["timing_sums"]["wall_total_observed"]
            - float(previous_timing.get("wall_total_observed", 0.0)),
        }
        completed = current["completed"] - int(previous_snapshot.get("completed", 0))
        calls = current["step7_calls"] - int(previous_snapshot.get("step7_calls", 0))
        if completed <= 0:
            return None
        return diagnostic_view_from_raw(
            batch_timing,
            _subtract_mapping(current["token_sums"], previous_snapshot.get("token_sums", {})),
            completed,
            calls,
        )

    batch_sums = _subtract_mapping(
        current["legacy_sums"], previous_snapshot.get("legacy_sums", {})
    )
    batch_counts = {
        key: int(value) - int(previous_snapshot.get("legacy_counts", {}).get(key, 0))
        for key, value in current["legacy_counts"].items()
    }
    completed = cumulative.get("completed", 0) - int(previous_snapshot.get("completed", 0))
    if completed <= 0 or not _has_step_timing(cumulative):
        return None
    view = views_from_sums(
        batch_sums,
        batch_counts,
        current["step7_calls"] - int(previous_snapshot.get("step7_calls", 0)),
    )
    view.update(
        {
            "schema_version": 1,
            "conservation_guaranteed": False,
            "completed": completed,
        }
    )
    return view


def _has_step_timing(timings: dict[str, Any]) -> bool:
    return bool(timings.get("time")) or bool(timings.get("per_question")) or bool(
        timings.get("token_per_question")
    )


def _format_time_group(
    view: dict[str, Any], title: str, retry_mode: str
) -> list[str]:
    group = view["time"][retry_mode]
    retry_label = "含重试" if retry_mode == "with_retry" else "不含失败重试"
    lines = [f"\n⏱️ {title} - {retry_label} ({view['completed']} 个问题):", "=" * 50]
    lines.extend(
        f"  {index:02d}_{stage}: {group['stages'][stage]:.6f}s/问题"
        for index, stage in enumerate(SAG2_TIMING_STAGE_ORDER, start=1)
    )
    lines.append(f"  99_total: {group['total']:.6f}s/问题")
    lines.append("=" * 50)
    return lines


def _format_token_group(view: dict[str, Any], title: str) -> list[str]:
    token = view["token"]
    return [
        f"\n🔢 {title} ({token['completed']} 个问题, {token['calls']} 次调用):",
        "=" * 50,
        (
            "  sum: prompt={prompt:.0f}, completion={completion:.0f}, total={total:.0f}"
        ).format(**token["sum"]),
        (
            "  avg/question: prompt={prompt:.2f}, completion={completion:.2f}, "
            "total={total:.2f}"
        ).format(**token["avg_per_question"]),
        (
            "  avg/call: prompt={prompt:.2f}, completion={completion:.2f}, total={total:.2f}"
        ).format(**token["avg_per_call"]),
        "=" * 50,
    ]


def _format_supplementary_lines(
    cum: dict[str, Any], batch: dict[str, Any] | None
) -> list[str]:
    """Format v2 diagnostics in the same order used by MLflow keys."""

    if cum.get("schema_version") != 2:
        return _format_step_timing_lines(cum, "累积各 Step 平均耗时") or []
    lines = _format_time_group(cum, "累积耗时", "with_retry")
    if batch is not None:
        lines.extend(_format_time_group(batch, "本批耗时", "with_retry"))
    lines.extend(_format_time_group(cum, "累积耗时", "no_retry"))
    if batch is not None:
        lines.extend(_format_time_group(batch, "本批耗时", "no_retry"))
    # Token is intentionally last.
    lines.extend(_format_token_group(cum, "累积 Token 统计"))
    if batch is not None:
        lines.extend(_format_token_group(batch, "本批 Token 统计"))
    return lines


def _format_step_timing_lines(
    timings: dict[str, Any], title: str, count_label: str = "个问题，累积平均"
) -> list[str] | None:
    """Legacy single-view formatter; schema v2 delegates to ordered groups."""

    if not _has_step_timing(timings):
        return None
    if timings.get("schema_version") == 2:
        lines = _format_time_group(timings, title, "with_retry")
        lines.extend(_format_time_group(timings, title, "no_retry"))
        lines.extend(_format_token_group(timings, f"{title} Token"))
        return lines

    pq = timings["per_question"]
    s7 = timings["step7_per_call"]
    n = timings.get("completed", "")
    tpq = timings.get("token_per_question", {})
    tpc = timings.get("token_per_call", {})
    lines = [f"\n⏱️ {title} ({n} {count_label}):", "=" * 50]
    lines.append(
        "  " + "  ".join(f"{k}: {pq.get(k, 0.0):.2f}s" for k in _STEP_DISPLAY_KEYS if k in pq)
    )
    if "step7_llm_no_retry" in pq:
        lines.append(
            f"  Step7-LLM(含重试): {s7['with_retry']:.2f}s/次  "
            f"Step7-LLM(不含重试): {s7['no_retry']:.2f}s/次  (共 {s7['calls']} 次)"
        )
    lines.append(
        f"  总耗时(Σstep平均, Step7不含重试): {timings['total_step7_no_retry']:.2f}s/问题"
    )
    if tpq:
        lines.append(
            f"  Step7-Token(每问题): 输入 {tpq.get('step7_prompt_tokens', 0.0):.1f}  "
            f"输出 {tpq.get('step7_completion_tokens', 0.0):.1f}"
        )
        lines.append(
            f"  Step7-Token(每次): 输入 {tpc.get('prompt', 0.0):.1f}  "
            f"输出 {tpc.get('completion', 0.0):.1f}  (共 {tpc.get('calls', 0)} 次)"
        )
    lines.append("=" * 50)
    return lines


def _append_time_metrics(
    metrics: dict[str, float], prefix: str, view: dict[str, Any], retry_mode: str
) -> None:
    group = view["time"][retry_mode]
    for index, stage in enumerate(SAG2_TIMING_STAGE_ORDER, start=1):
        metrics[f"{prefix}_{index:02d}_{stage}"] = float(group["stages"][stage])
    metrics[f"{prefix}_99_total"] = float(group["total"])


def _append_token_metrics(metrics: dict[str, float], prefix: str, view: dict[str, Any]) -> None:
    token = view["token"]
    for key in ("prompt", "completion", "total"):
        metrics[f"{prefix}_{key}_sum"] = float(token["sum"][key])
    for key in ("prompt", "completion", "total"):
        metrics[f"{prefix}_{key}_avg_per_question"] = float(token["avg_per_question"][key])
    for key in ("prompt", "completion", "total"):
        metrics[f"{prefix}_{key}_avg_per_call"] = float(token["avg_per_call"][key])
    metrics[f"{prefix}_calls"] = float(token["calls"])
    metrics[f"{prefix}_completed"] = float(token["completed"])


def build_supplementary_metrics(
    cum: dict[str, Any], batch: dict[str, Any] | None = None
) -> dict[str, float]:
    """Build ordered MLflow diagnostics; export cumulative timing only.

    Batch views remain available to console reporting and token diagnostics,
    but MLflow timing intentionally contains neither batch nor per-call views.
    """

    if cum.get("schema_version") != 2:
        return {}
    metrics: dict[str, float] = {}
    _append_time_metrics(metrics, "supp_01_time_cum_with_retry", cum, "with_retry")
    _append_time_metrics(metrics, "supp_02_time_cum_no_retry", cum, "no_retry")
    _append_token_metrics(metrics, "supp_03_token_cum", cum)
    if batch is not None:
        _append_token_metrics(metrics, "supp_04_token_batch", batch)
    return metrics


def public_search_diagnostics(view: dict[str, Any]) -> dict[str, Any]:
    """Drop raw accumulator and legacy compatibility fields for JSON output."""

    if view.get("schema_version") != 2:
        return {
            "schema_version": view.get("schema_version", 1),
            "conservation_guaranteed": False,
            "completed": view.get("completed", 0),
            "legacy_timings": view.get("per_question", {}),
        }
    return {
        "schema_version": 2,
        "conservation_guaranteed": bool(view.get("conservation_guaranteed")),
        "completed": view.get("completed", 0),
        "time": view["time"],
        "token": view["token"],
    }


def _build_timing_mlflow_metrics(timings: dict[str, Any], suffix: str = "") -> dict[str, float]:
    """Flatten cumulative legacy timing without per-call metrics."""

    flat = dict(timings.get("per_question", {}))
    flat["total_step7_no_retry"] = timings["total_step7_no_retry"]
    return {f"{k}{suffix}": v for k, v in flat.items()}


def _build_token_mlflow_metrics(timings: dict[str, Any], suffix: str = "") -> dict[str, float]:
    """Legacy MLflow token flattener retained for non-v2 callers."""

    tpq = timings.get("token_per_question", {})
    tpc = timings.get("token_per_call", {})
    metrics: dict[str, float] = {}
    if "step7_prompt_tokens" in tpq:
        metrics[f"step7_prompt{suffix}"] = tpq["step7_prompt_tokens"]
    if "step7_completion_tokens" in tpq:
        metrics[f"step7_completion{suffix}"] = tpq["step7_completion_tokens"]
    metrics[f"step7_prompt_percall{suffix}"] = tpc.get("prompt", 0.0)
    metrics[f"step7_completion_percall{suffix}"] = tpc.get("completion", 0.0)
    return metrics
