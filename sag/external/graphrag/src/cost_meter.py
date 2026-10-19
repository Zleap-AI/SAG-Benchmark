# Copyright (c) 2025.
"""轻量成本计数器：统计 GraphRAG 各阶段的 token 消耗与 API 调用次数。

用于复现 LightRAG Table 2 那种"检索阶段 / 增量更新阶段的 token & API 成本"对比。

用法（在跑 index / update / query 前设两个环境变量）：
    export GRAPHRAG_COST_PHASE=index          # 阶段标签：index | update | query
    export GRAPHRAG_COST_FILE=/abs/path/cost.json   # 结果落盘路径（追加式累加）

埋点位置：graphrag/language_model/providers/fnllm/models.py 的
    - OpenAIChatFNLLM.achat            （所有 chat 调用的唯一收口）
    - OpenAIEmbeddingFNLLM.aembed*     （所有 embedding 调用的收口）

结果：进程退出时（或手动 flush 时）写入 GRAPHRAG_COST_FILE，形如
    {
      "index":  {"chat_calls": N, "chat_input_tokens": ..., "chat_output_tokens": ...,
                 "embed_calls": M, "embed_texts": ...},
      "query":  {...},
      "update": {...}
    }
若文件已存在则读入后累加（方便多次 query 累计），所以每组新实验前先删旧文件。
"""

from __future__ import annotations

import atexit
import json
import os
import threading

_lock = threading.Lock()
_data: dict[str, dict[str, int]] = {}
_dumped = False


def _phase() -> str:
    return os.getenv("GRAPHRAG_COST_PHASE", "unknown")


def _bucket(phase: str) -> dict[str, int]:
    b = _data.get(phase)
    if b is None:
        b = {
            "chat_calls": 0,
            "chat_input_tokens": 0,
            "chat_output_tokens": 0,
            "embed_calls": 0,
            "embed_tokens": 0,
            "elapsed_s": 0.0,
        }
        _data[phase] = b
    return b


def record_chat(input_tokens: int, output_tokens: int, cache_hit: bool = False) -> None:
    """记一次 chat 调用。cache_hit=True 时不计 API 调用与 token（命中缓存没真发请求）。"""
    if cache_hit:
        return
    with _lock:
        b = _bucket(_phase())
        b["chat_calls"] += 1
        b["chat_input_tokens"] += int(input_tokens or 0)
        b["chat_output_tokens"] += int(output_tokens or 0)


def record_embed(input_tokens: int, cache_hit: bool = False) -> None:
    """记一次 embedding 调用。input_tokens=本次编码文本的 token 总数。"""
    if cache_hit:
        return
    with _lock:
        b = _bucket(_phase())
        b["embed_calls"] += 1
        b["embed_tokens"] += int(input_tokens or 0)


def record_elapsed(seconds: float) -> None:
    """记当前阶段的耗时。"""
    with _lock:
        b = _bucket(_phase())
        b["elapsed_s"] += float(seconds)


def dump() -> None:
    """把累计结果写入 GRAPHRAG_COST_FILE（与已有文件累加）。"""
    path = os.getenv("GRAPHRAG_COST_FILE")
    if not path:
        return
    with _lock:
        merged: dict[str, dict[str, int]] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    merged = json.load(f)
            except Exception:
                merged = {}
        for phase, b in _data.items():
            tgt = merged.setdefault(phase, {})
            for k, v in b.items():
                tgt[k] = tgt.get(k, 0) + v
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        # 落盘后清空内存计数，避免同进程内重复 dump 时二次累加
        _data.clear()


@atexit.register
def _flush_on_exit() -> None:
    dump()


_enc = None


def count_tokens(texts: list[str]) -> int:
    """用 cl100k_base 估算文本 token 数（embedding 服务端不返回 usage 时的兜底）。"""
    global _enc
    try:
        if _enc is None:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(_enc.encode(t or "")) for t in texts)
    except Exception:  # noqa: BLE001  兜底失败就退回字符/4 粗估
        return sum(len(t or "") for t in texts) // 4


_bge_tok = None
_bge_tok_failed = False


def _get_bge_tokenizer():
    """加载 bge 自己的 tokenizer 做精确截断（路径由 GRAPHRAG_EMBED_TOKENIZER 指定）。"""
    global _bge_tok, _bge_tok_failed
    if _bge_tok is not None or _bge_tok_failed:
        return _bge_tok
    path = os.getenv("GRAPHRAG_EMBED_TOKENIZER")
    if not path:
        _bge_tok_failed = True
        return None
    try:
        from transformers import AutoTokenizer

        _bge_tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:  # noqa: BLE001  加载失败就退回 cl100k
        _bge_tok_failed = True
        _bge_tok = None
    return _bge_tok


def truncate_texts(texts: list[str], max_tokens: int) -> list[str]:
    """把每条文本截断到 max_tokens，用于 embedding 前防止超过服务端上限（bge 512）。

    优先用 bge 真实 tokenizer（GRAPHRAG_EMBED_TOKENIZER 指定路径）精确截断，
    这样 max_tokens 就是 bge 端的真实 token 数；无该 tokenizer 时退回 cl100k 估算。
    行为等价于 HippoRAG 的 embedding_max_seq_len 截断。
    """
    # 预留 special tokens 余量（bge encode 会加 [CLS]/[SEP]），留 2 个
    budget = max(1, max_tokens - 2)

    bge = _get_bge_tokenizer()
    if bge is not None:
        out = []
        for t in texts:
            ids = bge.encode(t or "", add_special_tokens=False)
            if len(ids) > budget:
                out.append(bge.decode(ids[:budget]))
            else:
                out.append(t or "")
        return out

    # 退回 cl100k 估算（无 bge tokenizer 时）
    global _enc
    try:
        if _enc is None:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        out = []
        for t in texts:
            ids = _enc.encode(t or "")
            if len(ids) > budget:
                out.append(_enc.decode(ids[:budget]))
            else:
                out.append(t or "")
        return out
    except Exception:  # noqa: BLE001  失败就按字符粗截（4 char≈1 token）
        return [(t or "")[: budget * 4] for t in texts]

