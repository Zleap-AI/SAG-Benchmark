"""Request-local LLM and stage timing service for SAG2."""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from typing import Any

from pipeline.modules.search.sag2.contracts import SAG2_TIMING_STAGE_ORDER
from pipeline.modules.search.sag2.utils import round_seconds


class SAG2TimingService:
    """Own timing state while remaining reusable across concurrent requests."""

    def __init__(self, runtime: Any, *, ctx_var_name: str) -> None:
        self._runtime = runtime
        self._ctx: ContextVar[dict[str, float | int] | None] = ContextVar(
            ctx_var_name, default=None
        )

    @property
    def context_var(self) -> ContextVar[dict[str, float | int] | None]:
        return self._ctx

    def record_llm_call_metrics(self) -> None:
        acc = self._ctx.get()
        if acc is None:
            return
        llm_client = self._runtime.llm_client
        if llm_client is None:
            return
        timing = getattr(llm_client, "_last_call_timing", None)
        if not timing:
            return
        acc["with_retry"] += timing.get("total_time", 0.0)
        acc["no_retry"] += timing.get("success_time", 0.0)
        acc["wasted_retry_time"] = acc.get("wasted_retry_time", 0.0) + timing.get(
            "wasted_retry_time", 0.0
        )
        acc["calls"] += 1
        usage = getattr(llm_client, "_last_call_usage", None)
        if usage is not None:
            acc["prompt_tokens"] = acc.get("prompt_tokens", 0) + int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            acc["completion_tokens"] = acc.get("completion_tokens", 0) + int(
                getattr(usage, "completion_tokens", 0) or 0
            )

    def retry_wasted_snapshot(self) -> float:
        timing = self._ctx.get() or {}
        return float(timing.get("wasted_retry_time", 0.0) or 0.0)

    def start_request(self) -> Token:
        return self._ctx.set(
            {
                "with_retry": 0.0,
                "no_retry": 0.0,
                "wasted_retry_time": 0.0,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        )

    @staticmethod
    def record_timed_stage(
        stages_with_retry: dict[str, float],
        retry_wasted_by_stage: dict[str, float],
        stage: str,
        started_at: float,
        retry_wasted_before: float = 0.0,
        retry_wasted_after: float | None = None,
    ) -> None:
        wall = max(0.0, time.perf_counter() - started_at)
        after = retry_wasted_before if retry_wasted_after is None else retry_wasted_after
        stages_with_retry[stage] = wall
        retry_wasted_by_stage[stage] = max(0.0, float(after) - float(retry_wasted_before))

    @classmethod
    def build_timing_payload(
        cls,
        stages_with_retry_raw: dict[str, float],
        retry_wasted_raw: dict[str, float],
        wall_total_observed: float,
        step7_timing: dict[str, float | int],
    ) -> dict[str, Any]:
        with_retry: dict[str, float] = {}
        no_retry: dict[str, float] = {}
        retry_wasted: dict[str, float] = {}
        for stage in SAG2_TIMING_STAGE_ORDER:
            with_value = round_seconds(max(0.0, stages_with_retry_raw.get(stage, 0.0)))
            wasted_value = round_seconds(
                min(with_value, max(0.0, retry_wasted_raw.get(stage, 0.0)))
            )
            no_value = round_seconds(max(0.0, with_value - wasted_value))
            wasted_value = with_value - no_value
            with_retry[stage] = with_value
            no_retry[stage] = no_value
            retry_wasted[stage] = wasted_value

        total_with_retry = sum(with_retry.values())
        total_no_retry = sum(no_retry.values())
        retry_wasted_total = total_with_retry - total_no_retry
        payload: dict[str, Any] = {
            "schema_version": 2,
            "stage_order": list(SAG2_TIMING_STAGE_ORDER),
            "stages_with_retry": with_retry,
            "stages_no_retry": no_retry,
            "retry_wasted_by_stage": retry_wasted,
            "total_with_retry": total_with_retry,
            "total_no_retry": total_no_retry,
            "retry_wasted_total": retry_wasted_total,
            "wall_total_observed": round_seconds(max(0.0, wall_total_observed)),
            "step7_llm_with_retry": round_seconds(
                float(step7_timing.get("with_retry", 0.0) or 0.0)
            ),
            "step7_llm_no_retry": round_seconds(float(step7_timing.get("no_retry", 0.0) or 0.0)),
            "step7_llm_calls": int(step7_timing.get("calls", 0) or 0),
            "step7_prompt_tokens": int(step7_timing.get("prompt_tokens", 0) or 0),
            "step7_completion_tokens": int(step7_timing.get("completion_tokens", 0) or 0),
        }
        payload.update(with_retry)
        payload["total"] = total_with_retry
        cls.validate_timing_payload(payload)
        return payload

    @staticmethod
    def validate_timing_payload(payload: dict[str, Any]) -> None:
        tolerance = 1e-6
        with_total = sum(payload["stages_with_retry"].values())
        no_total = sum(payload["stages_no_retry"].values())
        if abs(with_total - payload["total_with_retry"]) > tolerance:
            raise ValueError("SAG2 timing v2 with-retry stages do not conserve total")
        if abs(no_total - payload["total_no_retry"]) > tolerance:
            raise ValueError("SAG2 timing v2 no-retry stages do not conserve total")
        if (
            abs(
                payload["total_with_retry"]
                - payload["total_no_retry"]
                - payload["retry_wasted_total"]
            )
            > tolerance
        ):
            raise ValueError("SAG2 timing v2 retry waste does not conserve totals")

    @staticmethod
    def build_event_stats(
        *,
        query_events: list[Any],
        entity_events: Any,
        expand_event_ids: list[str],
        event_ids: list[str],
        score_top_event_ids: list[str],
        rank_event_ids: list[str],
        seen_event_ids: set[str],
        query_entity_ids: list[str],
        new_entity_ids: list[str],
        scope: Any,
    ) -> dict[str, int]:
        return {
            "query_event_count": len(query_events),
            "entity_event_count": len(entity_events),
            "expand_event_count": len(expand_event_ids),
            "merged_event_count": len(event_ids),
            "score_top_event_count": len(score_top_event_ids),
            "rank_event_count": len(rank_event_ids),
            "seen_event_count": len(seen_event_ids),
            "candidate_scope_event_count": len(scope.event_scores) if scope is not None else 0,
            "candidate_scope_entity_count": len(scope.entity_ids) if scope is not None else 0,
            "candidate_scope_edge_count": (
                sum(len(values) for values in scope.event_to_entities.values())
                if scope is not None
                else 0
            ),
            "query_entity_count": len(query_entity_ids),
            "new_entity_count": len(new_entity_ids),
        }


__all__ = ["SAG2TimingService"]
