"""Embedding 维度动态解析（probe + 隔离缓存）

设计要点：
- OpenAI 兼容端点没有"查询维度"的标准 API，唯一可靠手段是 embed 一段短文本读长度。
- 缓存按 (base_url, model, requested_dimensions) 三元组隔离，换任一项都不串味。
- L1 进程内 dict（同进程只 probe 一次）+ L2 落盘 JSON（跨进程/跨次运行复用，TTL 7 天）。
- EMBEDDING_DIM 不是权威值，只是校验基准与离线 fallback。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline.core.config import get_settings
from pipeline.exceptions import AIError
from pipeline.utils import get_logger

logger = get_logger("ai.embedding_dim")

CACHE_TTL = timedelta(days=7)
PROBE_TEXT = "dimension probe"

# L1：进程内缓存 {cache_key: dim}
_MEM_CACHE: dict[str, int] = {}


class EmbeddingDimMismatchError(AIError):
    """probe 结果与 EMBEDDING_DIM 期望值不符（strict 模式下抛出）"""


def _project_root() -> Path:
    # 本文件位于 pipeline/core/ai/ 下，回退四级到项目根
    return Path(__file__).resolve().parent.parent.parent.parent


def _cache_path() -> Path:
    return _project_root() / ".cache" / "embedding_dims.json"


def make_cache_key(base_url: str | None, model: str, requested_dimensions: int | None) -> str:
    """隔离缓存 key：三元组任一变化即视为不同的 embedding 空间。

    刻意不含 api_key —— 同端点换 key 不会改变输出维度，且避免把密钥写进缓存文件。
    """
    raw = f"{base_url or ''}|{model}|{requested_dimensions if requested_dimensions else '-'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_disk_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"embedding 维度缓存读取失败（忽略）: {e}")
        return {}


def _save_disk_cache(cache: dict[str, Any]) -> None:
    """原子写：同目录临时文件 + os.replace，避免多进程并发写坏 JSON。"""
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".embedding_dims-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as e:
        logger.warning(f"embedding 维度缓存写入失败（不影响本次运行）: {e}")


def _entry_is_fresh(entry: dict[str, Any]) -> bool:
    probed_at = entry.get("probed_at")
    if not probed_at:
        return False
    try:
        ts = datetime.fromisoformat(probed_at)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < CACHE_TTL


def reset_embedding_dim_cache() -> None:
    """清空 L1（测试用）。不动 L2 落盘缓存。"""
    _MEM_CACHE.clear()


async def resolve_embedding_dim(
    *,
    force_probe: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """解析当前 embedding 配置的真实向量维度。

    Returns:
        {
          "dim": int,
          "dim_source": "cache" | "probe" | "stale_cache" | "env",
          "expected_dim": int | None,     # EMBEDDING_DIM
          "model": str,
          "base_url": str | None,
          "requested_dimensions": int | None,
          "probed_at": str | None,
          "cache_key": str,
        }

    Raises:
        EmbeddingDimMismatchError: strict 模式且 probe 与 EMBEDDING_DIM 不符
        AIError: probe 失败且无任何 fallback
    """
    settings = get_settings()
    model = settings.embedding_model_name
    base_url = settings.embedding_base_url or settings.llm_base_url
    requested = settings.embedding_request_dimensions
    expected = settings.embedding_dimensions
    force_probe = force_probe or os.getenv("EMBEDDING_DIM_FORCE_PROBE", "").lower() in (
        "1", "true", "yes",
    )

    key = make_cache_key(base_url, model, requested)
    base_result = {
        "model": model,
        "base_url": base_url,
        "requested_dimensions": requested,
        "expected_dim": expected,
        "cache_key": key,
    }

    # L1
    if not force_probe and key in _MEM_CACHE:
        return {**base_result, "dim": _MEM_CACHE[key], "dim_source": "cache", "probed_at": None}

    # L2
    disk = _load_disk_cache()
    entry = disk.get(key)
    if not force_probe and entry and _entry_is_fresh(entry) and isinstance(entry.get("dim"), int):
        _MEM_CACHE[key] = entry["dim"]
        logger.info(f"命中 embedding 维度缓存: dim={entry['dim']} (key={key})")
        return {
            **base_result,
            "dim": entry["dim"],
            "dim_source": "cache",
            "probed_at": entry.get("probed_at"),
        }

    # L3 probe
    try:
        if client is None:
            from pipeline.core.ai.factory import get_embedding_client

            client = await get_embedding_client(scenario="general")
        dim = await client.probe_dimensions(PROBE_TEXT)
    except Exception as e:  # noqa: BLE001 —— probe 失败必须走 fallback，不能中断 benchmark
        logger.warning(f"embedding 维度 probe 失败: {type(e).__name__}: {e}")
        if entry and isinstance(entry.get("dim"), int):
            logger.warning(f"⚠️  使用过期缓存维度 dim={entry['dim']}（probe 不可用）")
            _MEM_CACHE[key] = entry["dim"]
            return {
                **base_result,
                "dim": entry["dim"],
                "dim_source": "stale_cache",
                "probed_at": entry.get("probed_at"),
            }
        if expected:
            logger.warning(f"⚠️  回退到 EMBEDDING_DIM={expected}（probe 不可用且无缓存）")
            _MEM_CACHE[key] = expected
            return {**base_result, "dim": expected, "dim_source": "env", "probed_at": None}
        raise AIError(
            "无法确定 embedding 维度：probe 失败、无缓存、且未配置 EMBEDDING_DIM。"
            f" model={model} base_url={base_url}"
        ) from e

    # 与期望值比对
    if expected is not None and expected != dim:
        msg = (
            f"❌ embedding 维度不一致：服务端实际 dim={dim}，"
            f"但 EMBEDDING_DIM={expected}（model={model}, base_url={base_url}）"
        )
        if settings.embedding_dim_strict:
            raise EmbeddingDimMismatchError(msg)
        logger.error(msg + " —— 以服务端实际值为准，请修正 .env")

    probed_at = datetime.now(timezone.utc).isoformat()
    _MEM_CACHE[key] = dim
    disk[key] = {
        "base_url": base_url,
        "model": model,
        "requested_dimensions": requested,
        "dim": dim,
        "probed_at": probed_at,
    }
    _save_disk_cache(disk)
    logger.info(f"✅ embedding 维度 probe: dim={dim} (model={model}, key={key})")
    return {**base_result, "dim": dim, "dim_source": "probe", "probed_at": probed_at}
