"""轻量成本计数器：统计 HippoRAG2 各阶段（index/retrieval/qa）的 LLM token 消耗与耗时。

与 sag-benchmark/graphrag 项目的三阶段 cost/speed 统计口径对齐（见
/home/intern/mly/cost_speed_spec.md），产出同构的 cost.json，可直接拼进同一张对比表。

用法：
    from src.hipporag.utils import cost_tracker

    cost_tracker.set_output_path("outputs/musique/qwen.../cost.json")

    with cost_tracker.phase("index"):
        ...  # 阶段内所有经过 CacheOpenAI.infer() 的 chat 调用都会被记到 "index" 桶

进程退出时（atexit）自动落盘到 set_output_path 指定的文件，与已有内容合并累加
（同一进程内重复 dump 会清空内存避免二次累加；跨进程多次运行会与旧文件累加，
如需重新统计请先删除 cost.json）。

埋点位置：src/hipporag/llm/openai_gpt.py 的 CacheOpenAI.infer()。
该方法被 @cache_response 装饰，缓存命中时方法体不会执行，故 record_chat()
天然只统计真实发生的 API 调用，无需显式判断 cache_hit。

阶段边界：src/hipporag/HippoRAG.py 的 index() / retrieve() / qa() 三个方法体
各用 `with cost_tracker.phase(...)` 包裹。三个阶段在单进程内严格顺序执行、
从不并发，即使 index 阶段内部用 ThreadPoolExecutor 起多个 worker 线程调用
infer()，用模块级全局变量记录"当前阶段"也是安全的（无需 threading.local）。
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_lock = threading.Lock()
_data: dict[str, dict[str, float]] = {}
_current_phase: Optional[str] = None
_output_path: Optional[str] = None


def set_output_path(path: str) -> None:
    """设置 cost.json 的落盘路径。"""
    global _output_path
    _output_path = path


def _bucket(phase_name: str) -> dict[str, float]:
    b = _data.get(phase_name)
    if b is None:
        b = {
            "chat_calls": 0,
            "chat_input_tokens": 0,
            "chat_output_tokens": 0,
            "elapsed_s": 0.0,
        }
        _data[phase_name] = b
    return b


@contextmanager
def phase(name: str) -> Iterator[None]:
    """标记当前阶段并计时；退出时把耗时累加进该阶段的 elapsed_s。"""
    global _current_phase
    prev = _current_phase
    _current_phase = name
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        with _lock:
            _bucket(name)["elapsed_s"] += elapsed
        _current_phase = prev


def record_chat(prompt_tokens: int, completion_tokens: int) -> None:
    """记一次真实发生的 chat 调用（缓存命中不会触发这里，见模块docstring）。"""
    phase_name = _current_phase
    if phase_name is None:
        # 阶段外的调用（理论上不应发生），防御性跳过而不报错，避免影响主流程
        return
    with _lock:
        b = _bucket(phase_name)
        b["chat_calls"] += 1
        b["chat_input_tokens"] += int(prompt_tokens or 0)
        b["chat_output_tokens"] += int(completion_tokens or 0)


def dump() -> None:
    """把累计结果写入 _output_path，与已有文件内容合并累加。"""
    path = _output_path
    if not path:
        return
    with _lock:
        if not _data:
            return
        merged: dict[str, dict[str, float]] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    merged = json.load(f)
            except Exception:
                merged = {}
        for phase_name, b in _data.items():
            tgt = merged.setdefault(phase_name, {})
            for k, v in b.items():
                tgt[k] = tgt.get(k, 0) + v
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        # 落盘后清空内存计数，避免同进程内重复 dump 时二次累加
        _data.clear()


@atexit.register
def _flush_on_exit() -> None:
    dump()
