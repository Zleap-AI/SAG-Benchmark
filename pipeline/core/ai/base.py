"""
LLM客户端基类

定义LLM客户端的统一接口
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from pipeline.core.ai.models import LLMMessage, LLMResponse, ModelConfig
from pipeline.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMTransientError,
)
from pipeline.utils import get_logger

logger = get_logger("ai.llm")


# ---------------------------------------------------------------------------
# 结构化输出辅助（instructor 风格 chat_parsed 专用）
#
# OpenAI 的 response_format={"type":"json_schema", "strict": True} 要求 schema
# 满足 strict 约束：所有 object 递归 additionalProperties:false、所有可选字段进
# required、且不能含 title 等多余键。pydantic v2 的 model_json_schema() 默认不满足，
# 故先经 _to_strict_json_schema 规范化后再下发给模型。
# ---------------------------------------------------------------------------


def _extract_json_text(content: str) -> str:
    """从 LLM 返回内容中剥离 ```json 代码块，得到纯 JSON 文本。

    chat_with_schema 与 chat_parsed 共用，避免重复实现 markdown 剥离逻辑。
    """
    import re

    text = content.strip()
    json_block_match = re.search(
        r"```(?:json)?\s*\n(.*?)\n```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if json_block_match:
        logger.debug("从 markdown 代码块中提取 JSON")
        return json_block_match.group(1).strip()
    logger.debug("直接解析 JSON（无代码块）")
    return text


# OpenAI strict json_schema 要移除的多余键
_STRICT_STRIP_KEYS = ("title", "$defs", "default")


def _normalize_strict_schema(node: Any) -> Any:
    """递归规范化 JSON Schema 节点，使其满足 OpenAI strict 约束。

    - 移除 title / $defs / default 等多余键；
    - 为含 properties 的 object 节点补 additionalProperties: False；
    - 不支持嵌套 $ref（目标模型均为扁平结构）。
    """
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRICT_STRIP_KEYS:
                continue
            # 若存在 $ref，说明 schema 含嵌套引用，当前不支持
            if key == "$ref":
                raise LLMError(
                    "chat_parsed 暂不支持含嵌套模型($ref)的 pydantic 模型，请使用扁平字段结构"
                )
            cleaned[key] = _normalize_strict_schema(value)

        # object 且声明了 properties → strict 要求 additionalProperties=False
        if cleaned.get("type") == "object" and "properties" in cleaned:
            cleaned["additionalProperties"] = False
            # strict 要求所有字段必现：required 覆盖全部属性
            cleaned["required"] = list(cleaned["properties"].keys())

        return cleaned

    if isinstance(node, list):
        return [_normalize_strict_schema(item) for item in node]

    return node


def _to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """把 pydantic 模型类转成 OpenAI strict 兼容的 JSON Schema dict。"""
    raw_schema = model.model_json_schema()
    return _normalize_strict_schema(raw_schema)  # type: ignore[return-value]


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, config: ModelConfig) -> None:
        """
        初始化LLM客户端

        Args:
            config: LLM配置
        """
        self.config = config
        logger.info(
            "初始化%s客户端",
            config.provider.value,
            extra={"model": config.model},
        )

    async def _record_usage(self, usage: Any) -> None:
        """Publish response usage to the tracker active in this async context."""
        from pipeline.utils.llm_tracking import record_llm_usage

        await record_llm_usage(usage)

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            LLM响应

        Raises:
            LLMError: LLM调用失败
            LLMTimeoutError: 调用超时
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """
        流式聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            include_reasoning: 是否返回推理内容（reasoning_content）
            **kwargs: 其他参数

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）

        Raises:
            LLMError: LLM调用失败
            LLMTimeoutError: 调用超时
        """
        ...

    async def chat_with_schema(
        self,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        结构化输出（JSON Schema）

        注意：不自动注入提示词，调用方需在 messages 中自行定义输出格式要求。
        本方法只负责：1. 调用 LLM  2. 解析 JSON  3. Schema 校验（如果提供）

        Args:
            messages: 消息列表（应包含 SYSTEM 定义的输出格式）
            response_schema: JSON Schema定义（用于校验，可选）
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            解析后的JSON对象

        Raises:
            LLMError: LLM调用失败或JSON格式无效
            ValidationError: 响应不符合Schema（仅当提供schema时）
        """
        import json

        api_kwargs = dict(kwargs)
        if response_schema:
            api_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": response_schema,
                },
            }

        # 直接调用 LLM，不注入额外提示词
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **api_kwargs,
        )

        # 解析JSON响应
        try:
            import json

            # 提取JSON内容（可能被markdown代码块包裹）
            content = _extract_json_text(response.content)

            # 解析：先 json.loads，失败则用 json_repair 兜底（尾逗号、多余文字等）
            try:
                result = json.loads(content)
            except json.JSONDecodeError as parse_err:
                try:
                    import json_repair

                    result = json_repair.loads(content)
                    logger.info("LLM 返回的 JSON 经 json_repair 修复后解析成功")
                except Exception:
                    raise parse_err

            # 某些模型会返回“JSON字符串包裹JSON对象”，这里做一次解包。
            if isinstance(result, str):
                nested_content = result.strip()
                if nested_content.startswith(("{", "[")):
                    try:
                        result = json.loads(nested_content)
                    except json.JSONDecodeError:
                        try:
                            import json_repair

                            result = json_repair.loads(nested_content)
                            logger.info("LLM 返回了嵌套 JSON 字符串，已二次解析成功")
                        except Exception:
                            pass

            # 如果提供了schema，进行验证
            if response_schema:
                expected_type = response_schema.get("type")
                if expected_type == "object" and not isinstance(result, dict):
                    raise LLMError(
                        f"响应类型不符合Schema: 期望 object，实际 {type(result).__name__}"
                    )
                if expected_type == "array" and not isinstance(result, list):
                    raise LLMError(
                        f"响应类型不符合Schema: 期望 array，实际 {type(result).__name__}"
                    )

                # 尝试使用jsonschema进行严格验证
                try:
                    import jsonschema

                    jsonschema.validate(instance=result, schema=response_schema)
                    logger.debug("JSON schema validation passed")
                except ImportError:
                    # jsonschema未安装，使用简单验证（仅当根为 dict 时检查 required）
                    if isinstance(result, dict) and "properties" in response_schema:
                        required = response_schema.get("required", [])
                        for field in required:
                            if field not in result:
                                raise ValueError(f"缺少必需字段: {field}")
                    logger.debug("JSON simple validation passed")
                except Exception as e:
                    # jsonschema验证失败
                    if type(e).__name__ == "ValidationError":
                        logger.error(
                            "JSON schema validation failed: %s\n响应内容: %s",
                            e,
                            str(result)[:500],
                        )
                        raise LLMError(f"响应不符合Schema: {e}") from e
                    raise
            else:
                # 没有schema，只验证JSON格式（已通过json.loads）
                logger.debug("JSON format validation passed (no schema provided)")

            return result

        except json.JSONDecodeError as e:
            logger.error("JSON解析失败: %s\n内容: %s", e, response.content)
            raise LLMError(f"LLM返回的不是有效的JSON: {e}") from e
        except ValueError as e:
            logger.error("Schema验证失败: %s", e)
            raise LLMError(f"响应不符合Schema: {e}") from e

    async def chat_parsed(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        结构化输出（instructor 风格）

        直接接收 pydantic 模型类作为 response_format，由 client 内部完成
        「强制模型输出 JSON + 校验 + 反序列化」，返回模型实例（而非 dict）。

        实现要点：
        1. 将 response_model 经 _to_strict_json_schema 规范化为 OpenAI strict
           兼容的 JSON Schema，作为 response_format 从参数层强制模型输出合法 JSON；
        2. 返回内容剥离可能的 markdown 代码块后，用 response_model.model_validate_json
           反序列化为模型实例；解析失败抛 pydantic.ValidationError（由 LLMRetryClient
           决定是否重试）。

        Args:
            messages: 消息列表（应包含 SYSTEM 定义的输出格式说明）
            response_model: 期望输出的 pydantic 模型类
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数（透传给底层 chat）

        Returns:
            response_model 的实例

        Raises:
            ValidationError: 响应不符合模型 schema（可被 LLMRetryClient 重试）
            LLMError: LLM 调用失败或后端不支持 strict json_schema
        """
        strict_schema = _to_strict_json_schema(response_model)
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": strict_schema,
                    "strict": True,
                },
            },
            **kwargs,
        )

        content = _extract_json_text(response.content or "")
        # 暂存本次调用的 token 用量，供上层 Step7 统计（与 _last_call_timing 对称，
        # 只记「成功那次」——失败时 self.chat 抛异常、不会执行到这里）。
        self._last_call_usage = response.usage
        return response_model.model_validate_json(content)

    def _prepare_messages(
        self,
        messages: list[LLMMessage],
    ) -> list[dict[str, str]]:
        """
        准备消息列表（转换为API格式）

        Args:
            messages: 消息列表

        Returns:
            API格式的消息列表
        """
        return [msg.to_dict() for msg in messages]

    async def close(self) -> None:
        """
        关闭客户端，释放资源

        子类如果有需要关闭的资源（如HTTP连接），应该重写此方法
        """
        pass


class LLMRetryClient:
    """带重试机制的LLM客户端包装器"""

    def __init__(
        self,
        client: BaseLLMClient,
        max_retries: int | None = None,
        retry_delay: float = 4.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        初始化重试客户端

        Args:
            client: 基础LLM客户端
            max_retries: 最大重试次数（None则使用client配置）
            retry_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
        """
        self.client = client
        # Preserve the BaseLLMClient surface for callers that inspect model
        # metadata (for example Judge output naming).
        self.config = client.config
        self.max_retries = client.config.max_retries if max_retries is None else max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        # 最近一次 chat_parsed 调用的耗时口径（成功或失败都会刷新）。
        self._last_call_timing: dict[str, float | int | bool] = {}
        # 最近一次 chat_parsed 成功调用的 token 用量；失败调用始终为 None。
        self._last_call_usage: Any = None

    def _should_retry(self, error: Exception) -> bool:
        """
        判断错误是否应该重试

        Args:
            error: 异常对象

        Returns:
            True表示应该重试，False表示不应该重试
        """
        # 只重试明确的瞬态故障。参数、认证、响应校验和未知错误都必须立即上抛，
        # 以免把错误配置放大为多次相同请求。
        if isinstance(error, LLMRequestError):
            return False
        if isinstance(
            error,
            (
                LLMTimeoutError,
                LLMRateLimitError,
                LLMTransientError,
                asyncio.TimeoutError,
                ConnectionError,
            ),
        ):
            return True
        return False

    def _compute_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟（含随机抖动）

        delay = retry_delay × backoff_factor^attempt × (0.5 ~ 1.0 jitter)

        示例 (retry_delay=4, backoff_factor=2):
          attempt 0: 4 × 1  × jitter = 2.0~4.0s
          attempt 1: 4 × 2  × jitter = 4.0~8.0s
          attempt 2: 4 × 4  × jitter = 8.0~16.0s
          attempt 3: 4 × 8  × jitter = 16.0~32.0s
          attempt 4: 4 × 16 × jitter = 32.0~64.0s
        """
        base_delay = self.retry_delay * (self.backoff_factor**attempt)
        jitter = 0.5 + random.random() * 0.5
        return base_delay * jitter

    def _record_chat_parsed_failure(
        self,
        *,
        overall_t0: float,
        wasted_retry_time: float,
        retries: int,
    ) -> None:
        """Publish this failed logical call without retaining prior success slots."""

        self._last_call_timing = {
            "success_time": 0.0,
            "total_time": time.perf_counter() - overall_t0,
            "retries": retries,
            "wasted_retry_time": wasted_retry_time,
            "failed": True,
        }
        self._last_call_usage = None

    async def chat(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        带重试的聊天补全

        实现指数退避重试策略（含随机抖动，避免多 worker 同时重试）

        仅对明确的瞬态错误执行重试；不可恢复请求错误立即上抛。
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.chat(messages, **kwargs)
            except Exception as e:
                last_error = e

                # 判断是否应该重试
                if not self._should_retry(e):
                    logger.error("遇到不可重试错误: %s", e)
                    raise

                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "LLM调用失败，%.1fs后重试 (尝试 %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM调用失败，已重试%d次",
                        self.max_retries,
                        exc_info=True,
                    )

        raise LLMError(f"LLM调用失败，已重试{self.max_retries}次") from last_error

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """
        流式调用（不重试）

        流式调用失败时无法重试，直接抛出异常

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）
        """
        async for chunk in self.client.chat_stream(messages, **kwargs):
            yield chunk

    async def chat_with_schema(
        self,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        带重试的结构化输出

        根据错误类型智能决定是否重试
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.chat_with_schema(
                    messages,
                    response_schema,
                    **kwargs,
                )
            except Exception as e:
                last_error = e

                # 判断是否应该重试
                if not self._should_retry(e):
                    logger.error("遇到不可重试错误: %s", e)
                    raise

                if attempt < self.max_retries:
                    logger.warning(
                        "结构化输出失败，立即重试 (尝试 %d/%d)",
                        attempt + 1,
                        self.max_retries,
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )

        raise LLMError(
            f"结构化输出失败，已重试{self.max_retries}次, 最后一次错误: {last_error}"
        ) from last_error

    async def chat_parsed(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """
        带重试的结构化输出（instructor 风格）

        根据 _should_retry 智能决定是否重试：网络类错误（LLMError/LLMRateLimitError）
        与 pydantic.ValidationError 都会重试；LLMTimeoutError 不重试。

        额外记录耗时口径（便于区分「纯 LLM 计算时间」vs「重试浪费时间」）：
        - 每次 attempt 的墙钟耗时单独计时；
        - 成功后，若发生过重试，INFO 打印一行：总耗时 / 成功那次耗时 / 重试浪费耗时
          （= 失败 attempt 的耗时之和）/ 重试次数。上层 Step7 的端到端计时仍含这部分。
        """
        # Never expose metrics or usage from the preceding logical call while
        # this call is running or after it fails.
        self._last_call_timing = {}
        self._last_call_usage = None
        last_error: Exception | None = None
        wasted_retry_time = 0.0  # 失败 attempt 的墙钟耗时之和（浪费在重试上的时间）
        overall_t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            attempt_t0 = time.perf_counter()
            try:
                result = await self.client.chat_parsed(messages, response_model, **kwargs)
            except Exception as e:
                # 本次失败的耗时计入「浪费时间」
                wasted_retry_time += time.perf_counter() - attempt_t0
                last_error = e

                # 判断是否应该重试
                if not self._should_retry(e):
                    self._record_chat_parsed_failure(
                        overall_t0=overall_t0,
                        wasted_retry_time=wasted_retry_time,
                        retries=attempt,
                    )
                    logger.error("遇到不可重试错误: %s", e)
                    raise

                if attempt < self.max_retries:
                    logger.warning(
                        "结构化输出(chat_parsed)失败，立即重试 (尝试 %d/%d)",
                        attempt + 1,
                        self.max_retries,
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                continue

            # —— 成功 ——
            success_time = time.perf_counter() - attempt_t0
            # 记录本次调用双口径耗时供上层 Step7 统计：
            #   success_time = 成功那次墙钟（不含重试）
            #   total_time   = 整体重试墙钟（含重试；未重试时 == success_time）
            self._last_call_timing = {
                "success_time": success_time,
                "total_time": (time.perf_counter() - overall_t0) if attempt > 0 else success_time,
                "retries": attempt,
                "wasted_retry_time": wasted_retry_time,
                "failed": False,
            }
            # 从底层 client 转存本次「成功那次」的 token 用量（与 _last_call_timing 同源同时机，
            # 读取端统一读 LLMRetryClient 实例）。
            self._last_call_usage = getattr(self.client, "_last_call_usage", None)
            if attempt > 0:
                # 发生过重试才打点，区分纯计算 vs 重试浪费
                total_time = time.perf_counter() - overall_t0
                logger.info(
                    "[chat_parsed耗时] 总耗时=%.2fs, 成功那次=%.2fs, 重试浪费=%.2fs, 重试次数=%d",
                    total_time,
                    success_time,
                    wasted_retry_time,
                    attempt,
                )
            return result

        self._record_chat_parsed_failure(
            overall_t0=overall_t0,
            wasted_retry_time=wasted_retry_time,
            retries=self.max_retries,
        )
        raise LLMError(
            f"结构化输出(chat_parsed)失败，已重试{self.max_retries}次, 最后一次错误: {last_error}"
        ) from last_error

    async def close(self) -> None:
        """Close the wrapped client and release its transport resources."""
        await self.client.close()
