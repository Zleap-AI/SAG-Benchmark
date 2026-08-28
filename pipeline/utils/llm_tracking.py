"""Context-local LLM token tracking shared by upload, search, QA, and evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any


class LLMTokenTracker:
    """Accumulate LLM usage by explicit stage."""

    def __init__(self) -> None:
        self.total = {"prompt": 0, "completion": 0, "total": 0}
        self.stages: dict[str, dict[str, int]] = {}
        self._lock = asyncio.Lock()

    async def record(self, stage: str, usage: Any) -> None:
        if not usage:
            return

        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            total = int(usage.get("total_tokens", 0) or 0)
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            total = int(getattr(usage, "total_tokens", 0) or 0)

        if total <= 0:
            total = prompt + completion

        stage_name = (stage or "UNKNOWN").upper()
        async with self._lock:
            self.total["prompt"] += prompt
            self.total["completion"] += completion
            self.total["total"] += total

            stats = self.stages.setdefault(
                stage_name,
                {"calls": 0, "prompt": 0, "completion": 0, "total": 0},
            )
            stats["calls"] += 1
            stats["prompt"] += prompt
            stats["completion"] += completion
            stats["total"] += total

    def get_summary(self) -> dict[str, Any]:
        """Return the backward-compatible source_info.json usage shape.

        Callers should read the summary after all tracked tasks have completed.
        """
        return {
            "total_prompt": self.total["prompt"],
            "total_completion": self.total["completion"],
            "total_tokens": self.total["total"],
            "total_calls": sum(stage["calls"] for stage in self.stages.values()),
            "stages": {name: stats.copy() for name, stats in self.stages.items()},
        }


_ACTIVE_TRACKER: ContextVar[LLMTokenTracker | None] = ContextVar(
    "pipeline_active_llm_token_tracker",
    default=None,
)
_ACTIVE_STAGE: ContextVar[str] = ContextVar(
    "pipeline_active_llm_tracking_stage",
    default="UNKNOWN",
)
_VERBOSE_LLM: ContextVar[bool] = ContextVar(
    "pipeline_verbose_llm",
    default=False,
)


def enable_llm_verbose() -> None:
    """Enable verbose LLM logging for the current async context.

    When enabled, the full prompt (all messages) and full response content
    are logged at INFO level before and after every LLM call.
    """
    _VERBOSE_LLM.set(True)


def is_llm_verbose() -> bool:
    """Return whether verbose LLM logging is enabled in the current context."""
    return _VERBOSE_LLM.get()


@contextmanager
def llm_tracking_scope(token_tracker: LLMTokenTracker) -> Iterator[None]:
    """Activate a tracker only for the current context and its child tasks."""
    token = _ACTIVE_TRACKER.set(token_tracker)
    try:
        yield
    finally:
        _ACTIVE_TRACKER.reset(token)


@contextmanager
def llm_tracking_stage(stage: str) -> Iterator[None]:
    """Set an explicit tracking stage for the current context."""
    token = _ACTIVE_STAGE.set((stage or "UNKNOWN").upper())
    try:
        yield
    finally:
        _ACTIVE_STAGE.reset(token)


async def record_llm_usage(usage: Any) -> None:
    """Record usage in the tracker active for the current context, if any."""
    tracker = _ACTIVE_TRACKER.get()
    if tracker is not None and usage is not None:
        await tracker.record(_ACTIVE_STAGE.get(), usage)


def enable_llm_tracking(token_tracker: LLMTokenTracker) -> Callable[[], None]:
    """Compatibility API that activates context-local tracking.

    New code should prefer ``llm_tracking_scope``. This function deliberately
    does not mutate any LLM class or method.
    """
    token: Token[LLMTokenTracker | None] = _ACTIVE_TRACKER.set(token_tracker)
    restored = False

    def restore() -> None:
        nonlocal restored
        if not restored:
            _ACTIVE_TRACKER.reset(token)
            restored = True

    return restore


__all__ = [
    "LLMTokenTracker",
    "enable_llm_tracking",
    "enable_llm_verbose",
    "is_llm_verbose",
    "llm_tracking_scope",
    "llm_tracking_stage",
    "record_llm_usage",
]
