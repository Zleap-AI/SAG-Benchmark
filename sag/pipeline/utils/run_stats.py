"""通用运行阶段耗时与 LLM token 汇总工具。"""

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pipeline.utils.llm_tracking import LLMTokenTracker


@dataclass
class StageMeasurement:
    """单次阶段计时结果；退出 measure 上下文后可读取 elapsed_seconds。"""

    stage: str
    elapsed_seconds: float = 0.0


class RunStatsTracker:
    """统一记录阶段耗时与 LLM token，用于 upload/search/QA 等脚本。"""

    def __init__(self, token_tracker: LLMTokenTracker | None = None) -> None:
        self.token_tracker = token_tracker or LLMTokenTracker()
        self._started_at = time.perf_counter()
        self._stage_seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, stage: str) -> Iterator[StageMeasurement]:
        measurement = StageMeasurement(stage=stage)
        started_at = time.perf_counter()
        try:
            yield measurement
        finally:
            measurement.elapsed_seconds = time.perf_counter() - started_at
            self._stage_seconds[stage] += measurement.elapsed_seconds

    def add_duration(self, stage: str, seconds: float) -> None:
        self._stage_seconds[stage] += max(0.0, float(seconds))

    def token_snapshot(self) -> dict[str, int]:
        summary = self.token_tracker.get_summary()
        return {
            "input_tokens": summary["total_prompt"],
            "output_tokens": summary["total_completion"],
            "total_tokens": summary["total_tokens"],
            "calls": summary["total_calls"],
        }

    @staticmethod
    def token_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {
            key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
            for key in ("input_tokens", "output_tokens", "total_tokens", "calls")
        }

    def timing_summary(self) -> dict[str, Any]:
        stages = {
            name: {"seconds": round(seconds, 3)}
            for name, seconds in sorted(self._stage_seconds.items())
        }
        return {
            "total_seconds": round(time.perf_counter() - self._started_at, 3),
            "stages": stages,
        }

    def token_summary(self) -> dict[str, Any]:
        legacy = self.token_tracker.get_summary()
        return {
            "input_tokens": legacy["total_prompt"],
            "output_tokens": legacy["total_completion"],
            "total_tokens": legacy["total_tokens"],
            "calls": legacy["total_calls"],
            "stages": {
                name.lower(): {
                    "calls": values["calls"],
                    "input_tokens": values["prompt"],
                    "output_tokens": values["completion"],
                    "total_tokens": values["total"],
                }
                for name, values in legacy["stages"].items()
            },
        }

    def get_summary(self) -> dict[str, Any]:
        return {
            "timing": self.timing_summary(),
            "tokens": self.token_summary(),
        }


__all__ = ["RunStatsTracker", "StageMeasurement"]
