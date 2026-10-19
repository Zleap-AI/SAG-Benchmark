import functools
import gzip
import hashlib
import json
import os
import sqlite3
import zlib
from copy import deepcopy
from typing import List, Optional, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI, BadRequestError
from packaging import version

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from ..utils import cost_tracker
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def _text_garbage_ratio(text: str) -> float:
    """估计一段文本里"看起来像乱码"的字符占比。

    踩坑背景：一开始用的是"落在 U+0080-U+00FF 扩展 Latin-1 区间"来判定乱码，
    但实测线上真实的乱码样本（httpx 把没声明 Content-Encoding 的压缩响应体
    直接当文本硬解码出来的）字符散落在远比这宽的 Unicode 范围里（比如
    U+02CA、U+FFE2 这类 Spacing Modifier / Fullwidth 符号），落在 Latin-1
    区间的判定完全测不出来。改成反过来判断："正常错误信息该长什么样"——
    基本应该是 ASCII 可打印字符，偶尔夹一点常见中文（数字/汉字/中文标点）。
    不在这个白名单里的字符占比一高，就基本可以判定是乱码。
    """
    if not text:
        return 1.0
    good = 0
    for ch in text:
        code = ord(ch)
        if 0x20 <= code <= 0x7E:               # ASCII 可打印
            good += 1
        elif ch in "\n\t":                      # 常见空白符
            good += 1
        elif 0x4E00 <= code <= 0x9FFF:          # 常用汉字（CJK Unified Ideographs）
            good += 1
        elif ch in "，。、；：？！“”‘’（）《》【】…—·":  # 常见中文标点
            good += 1
    return 1.0 - good / len(text)


def _build_thinking_extra_body(base_url: Optional[str], enable_thinking: bool) -> dict:
    """按 LLM 端点构造关闭思维链的 extra_body。

    两种端点的思考模式传参方式不同：
      - 302ai / 网关代理：认顶层 enable_thinking
        extra_body = {"enable_thinking": False}
      - 本地 vLLM（OpenAI 兼容）：只认 chat_template_kwargs.enable_thinking
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    按 base_url 是否属于 302 网关（302ai / gpt.302.ai）分派；enable_thinking=True（保留思考）时不传任何
    关闭参数，交给模型默认行为。
    """
    if enable_thinking:
        return {}
    if base_url and ("302ai" in base_url.lower() or "gpt.302.ai" in base_url.lower()):
        return {"enable_thinking": False}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _try_recover_raw_body(e: Exception) -> Optional[str]:
    """response.text 解码出来的是乱码时，尝试对原始响应字节手动解压/解码一次。

    典型场景：代理返回的 body 实际上是 gzip/deflate 压缩过的，但响应头没有
    正确声明 Content-Encoding，httpx 就不会自动解压，直接把压缩后的原始字节
    当文本解码，产出满屏乱码。这里拿 `e.response.content`（httpx 已经按它自己
    判断处理过一遍的字节）分别尝试 gzip / zlib / raw-deflate 解压，能解出人类
    可读文本就用，解不出就返回 None，交给上层走乱码兜底提示。
    """
    response = getattr(e, "response", None)
    if response is None:
        return None
    try:
        raw = response.content
    except Exception:
        return None
    if not raw:
        return None

    for decompress in (gzip.decompress, zlib.decompress, lambda b: zlib.decompress(b, -zlib.MAX_WBITS)):
        try:
            decoded = decompress(raw).decode("utf-8")
        except Exception:
            continue
        if decoded.strip() and _text_garbage_ratio(decoded) < 0.1:
            return decoded
    return None


def _safe_error_summary(e: Exception, max_len: int = 300) -> str:
    """把 API 报错整理成人类可读的一行摘要。

    openai SDK 在构造 APIStatusError 时，如果响应体不是合法 JSON，会直接把
    `response.text`（httpx 按 Content-Type/猜测的字符集解码出来的字符串）塞进
    异常消息里。当代理返回的响应体实际是压缩数据、但响应头没有正确声明
    Content-Encoding 时，httpx 不会自动解压，直接把压缩字节硬解码成文本，
    产出一大段乱码——直接 `str(e)` 会把这堆乱码整段打进日志，完全看不出真实
    报错原因，还会污染日志文件（甚至因为混进控制字符让 grep 把日志当成二进制
    文件）。这里改成：
      1. body 是合法 JSON（dict）→ 直接展示，信息量最大
      2. body 是字符串但看着像乱码 → 尝试对原始响应字节手动解压，能解出人类
         可读文本就换上（见 _try_recover_raw_body）
      3. 解压也救不回来 → 明确提示"响应体不可读"，不把乱码写进日志
    """
    status = getattr(e, "status_code", "?")
    body = getattr(e, "body", None)

    if isinstance(body, dict):
        text = json.dumps(body, ensure_ascii=False)
    else:
        raw = str(body) if body is not None else str(e)
        if _text_garbage_ratio(raw) >= 0.1:
            recovered = _try_recover_raw_body(e)
            text = recovered.strip() if recovered else ""
            if not text:
                text = ("<响应体不是合法 JSON 且疑似乱码（可能是代理返回的压缩数据没有正确声明 "
                         "Content-Encoding，被当成纯文本硬解码），手动解压也未能恢复，故不打印原始内容>")
        else:
            # 基本可读，只是过滤掉零星的控制字符
            text = "".join(ch for ch in raw if ch.isprintable() or ch in "\n\t ").strip()
            if len(text) < 5:
                text = "<响应体为空或不可读>"

    if len(text) > max_len:
        text = text[:max_len] + "...(截断)"
    return f"status={status} {text}"

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, **config_dict)

    def __init__(self, cache_dir, cache_filename: str = None,
                 llm_name: str = "gpt-4o-mini", api_key: str = None, llm_base_url: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self._init_llm_config(**kwargs)
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None
        self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, http_client=client, max_retries=0)

    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "max_completion_tokens": kwargs.get("max_new_tokens", 400),
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.7),
            }
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        max_retries = 5

        # 关闭思维链（Qwen3.6 这类模型）：302ai 认顶层 enable_thinking，
        # 本地 vLLM 只认 chat_template_kwargs.enable_thinking（只传顶层会被忽略，
        # 模型思考把 max_tokens 预算烧光后返回 content=None）。按端点分派，
        # 统一读 LLM_ENABLE_THINKING（=1 保留思考）。
        extra_body = _build_thinking_extra_body(
            self.llm_base_url, os.getenv("LLM_ENABLE_THINKING", "0") == "1"
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = self.openai_client.chat.completions.create(**params, extra_body=extra_body)
                break
            except BadRequestError as e:
                # 400 是永久性错误（prompt 过长、内容违规等），重试无意义
                logger.warning(f"LLM BadRequest (400)，不再重试: {_safe_error_summary(e)}")
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning(f"LLM调用失败，正在重试 (尝试 {attempt}/{max_retries}): {_safe_error_summary(e)}")

        response_message = response.choices[0].message.content

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        # 记录真实 chat 调用到 cost_tracker（缓存命中由 cache_response 短路，不计数）
        cost_tracker.record_chat(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )

        return response_message, metadata


