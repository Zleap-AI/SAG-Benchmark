"""Step_1 索引进度的自定义 ProgressLogger —— 按 workflow 打印 已完成/总数 + 已用时 + ETA。

graphrag 内核的进度路径（已确认）：
  - derive_from_rows → progress_ticker(callbacks.progress, num_total=len(input)) 逐条 tick
  - run_pipeline → logger.child(name, transient=False)，workflow 结束时 progress(Progress(percent=1))
  - ProgressWorkflowCallbacks.workflow_start → self._latest.child(name)

所以 child(prefix) 收到的就是 workflow 名（extract_graph / create_community_reports …），
__call__(Progress) 收到 completed_items/total_items。

一个 workflow 内可能跑多个 ticker（如 extract_graph 先 50 个 chunk 抽取、再 924 条实体
汇总），每个 ticker 是一个独立「阶段」：按 total 区分，各自计时、各自收尾，否则 ETA
会用整条 pipeline 的已用时去外推当前阶段，算出离谱的剩余时间。

输出形如：
  [3/8 extract_graph]  22/50 (44.0%)  已用 01m00s  预计剩余 01m16s
  [3/8 extract_graph]  50/50 (100%)  用时 02m12s  ✓
"""

from __future__ import annotations

import time

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.logger.base import Progress, ProgressLogger


def _fmt_dur(seconds: float) -> str:
    """把秒格式化成 08m12s / 1h23m45s。"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


class ProgressState:
    """父子 logger 共享的状态。

    同一个 workflow 会有多个 logger 实例（run_pipeline 建一个、
    ProgressWorkflowCallbacks 再建一个），阶段计时与去重必须放在共享处。
    """

    def __init__(self) -> None:
        self.names: list[str] = []
        self.started: float = 0.0
        # (workflow, total) → 该阶段起始时刻
        self.phase_started: dict[tuple[str, int], float] = {}
        # 已打印收尾行的阶段
        self.phase_done: set[tuple[str, int]] = set()
        # 已打印收尾行的 workflow（percent=1 的整体结束信号）
        self.workflow_done: set[str] = set()


class StepProgressLogger(ProgressLogger):
    """按 workflow/阶段打印进度，节流输出（nohup 友好，默认 20s 一次）。"""

    def __init__(
        self,
        state: ProgressState | None = None,
        *,
        workflow: str | None = None,
        min_interval_s: float = 20.0,
    ) -> None:
        self._state = state or ProgressState()
        self._workflow = workflow
        self._min_interval_s = min_interval_s
        self._last_emit = 0.0

    @property
    def _prefix(self) -> str:
        names = self._state.names
        index = names.index(self._workflow) + 1 if self._workflow in names else 0
        return f"[{index}/{len(names)} {self._workflow}]"

    def __call__(self, update: Progress) -> None:
        if self._workflow is None or self._workflow not in self._state.names:
            return
        now = time.perf_counter()
        total = update.total_items
        completed = update.completed_items

        # 无逐条进度的信号（percent=1）：workflow 整体结束
        if total is None or completed is None:
            if update.percent is None or update.percent < 1.0:
                return
            if self._workflow in self._state.workflow_done:
                return
            self._state.workflow_done.add(self._workflow)
            # 该 workflow 已有阶段输出过明细时，不再补一行空泛的「完成」
            if any(wf == self._workflow for wf, _ in self._state.phase_done):
                return
            elapsed = now - self._state.started
            print(f"{self._prefix}  完成  用时 {_fmt_dur(elapsed)}  ✓", flush=True)
            return

        if total <= 0:
            return
        phase = (self._workflow, total)
        phase_start = self._state.phase_started.setdefault(phase, now)
        elapsed = now - phase_start

        if completed >= total:
            if phase in self._state.phase_done:
                return
            self._state.phase_done.add(phase)
            print(
                f"{self._prefix}  {completed}/{total} (100%)  用时 {_fmt_dur(elapsed)}  ✓",
                flush=True,
            )
            return

        if now - self._last_emit < self._min_interval_s:
            return
        self._last_emit = now
        pct = completed / total * 100
        eta = elapsed * (total - completed) / completed if completed else None
        eta_s = f"  预计剩余 {_fmt_dur(eta)}" if eta is not None else ""
        print(
            f"{self._prefix}  {completed}/{total} ({pct:.1f}%)  已用 {_fmt_dur(elapsed)}{eta_s}",
            flush=True,
        )

    def child(self, prefix: str, transient: bool = True) -> StepProgressLogger:
        return StepProgressLogger(self._state, workflow=prefix, min_interval_s=self._min_interval_s)

    def force_refresh(self) -> None:
        pass

    # ── 状态消息（root 实例收到；build_index 与 validate_config_names 都会调）──
    def info(self, message: str) -> None:
        # 每条 workflow 结束内核会 info(str(output.result))，是整张 DataFrame 的 repr，截断避免刷屏
        text = " ".join(str(message).split())
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"[graphrag] {text}", flush=True)

    def success(self, message: str) -> None:
        # build_index 对每个 workflow 调 success(workflow 名)，与阶段收尾行重复，跳过；
        # validate_config_names 的 "LLM/Embedding Config Params Validated" 正常打印。
        if message in self._state.names:
            return
        print(f"[graphrag] ✓ {message}", flush=True)

    def error(self, message: str) -> None:
        print(f"[graphrag] ✗ {message}", flush=True)

    def warning(self, message: str) -> None:
        print(f"[graphrag] ⚠ {message}", flush=True)

    def dispose(self) -> None:
        pass

    def stop(self) -> None:
        pass


class PipelineOutlineCallbacks(NoopWorkflowCallbacks):
    """只实现 pipeline_start：把 workflow 名单 + 起始时间写回共享 state，供 [i/N] 前缀使用。"""

    def __init__(self, state: ProgressState) -> None:
        self._state = state

    def pipeline_start(self, names: list[str]) -> None:
        self._state.names = list(names)
        self._state.started = time.perf_counter()
        print(
            f"[graphrag] 开始建索引：共 {len(names)} 个 workflow\n[graphrag]   {' → '.join(names)}",
            flush=True,
        )
