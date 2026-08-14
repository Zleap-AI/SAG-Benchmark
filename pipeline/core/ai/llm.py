"""
OpenAI LLM客户端实现

注意:
- 支持标准OpenAI模型 (sophnet/Qwen3-30B-A3B-Thinking-2507, gpt-3.5-turbo等)
- 支持思考模型 (Thinking Models): 某些模型(如Qwen3-30B-A3B-Thinking)会将推理过程
  放在reasoning_content字段中而不是content字段。本实现会自动检测并处理这种情况。
"""

from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from pipeline.core.ai.base import BaseLLMClient
from pipeline.core.ai.models import LLMMessage, LLMProvider, LLMResponse, LLMUsage, ModelConfig
from pipeline.core.config.settings import get_settings
from pipeline.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from pipeline.utils import get_logger

logger = get_logger("ai.openai")


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端实现"""

    def __init__(self, config: ModelConfig) -> None:
        """
        初始化OpenAI客户端

        Args:
            config: LLM配置
        """
        super().__init__(config)

        # 构建默认 headers（用于控制内容过滤等）
        default_headers = self._build_default_headers()

        # 创建AsyncOpenAI客户端
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            default_headers=default_headers if default_headers else None,
        )

    def _build_default_headers(self) -> dict:
        """
        构建默认请求头

        Returns:
            默认请求头字典
        """

        settings = get_settings()
        headers = {}

        # 如果禁用内容过滤（绿网），添加 DashScope header
        if not settings.llm_data_inspection:
            headers["X-DashScope-DataInspection"] = '{"input": "disable", "output": "disable"}'

        return headers

    def _build_extra_body(self) -> dict[str, Any]:
        """
        构建 chat.completions.create 的 extra_body。

        extra_body 用于透传 OpenAI 标准之外、但推理后端（vLLM/SGLang 等）支持的参数：
        - enable_thinking：思考模式开关（双路径 — 顶层给 302.ai，chat_template_kwargs 给 vLLM）
        - top_k / min_p / repetition_penalty：扩展采样参数（仅当非默认值时才下发，
          避免向不支持的后端发送冗余字段导致报错；OpenAI 官方会忽略这些键）

        Returns:
            extra_body 字典
        """
        extra_body: dict[str, Any] = {
            "enable_thinking": self.config.enable_thinking,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }

        # 扩展采样参数：默认值（top_k=-1 / min_p=0.0 / repetition_penalty=1.0）表示关闭，
        # 不下发，保持对其他后端的兼容性。
        if self.config.top_k != -1:
            extra_body["top_k"] = self.config.top_k
        if self.config.min_p:
            extra_body["min_p"] = self.config.min_p
        if self.config.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.config.repetition_penalty

        return extra_body

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        OpenAI聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            LLM响应

        Raises:
            LLMError: 调用失败
            LLMTimeoutError: 调用超时
            LLMRateLimitError: 速率限制
        """
        try:
            # 准备消息
            api_messages = self._prepare_messages(messages)
            # 记录使用的模型信息
            logger.info(
                "🤖 调用 LLM - 模型: %s, base_url: %s, temperature: %.2f, max_tokens: %s, timeout: %s, enable_think: %s",
                self.config.model,
                self.config.base_url,
                temperature or self.config.temperature,
                max_tokens or self.config.max_tokens or "未设置",
                self.config.timeout,
                self.config.enable_thinking,
            )

            # 记录消息规模（不打印内容，避免敏感/超长 prompt 刷屏）
            logger.debug(
                "📤 LLM 请求消息 %d 条, 总字符数 %d",
                len(messages),
                sum(len(m.content) for m in messages),
            )

            # 调用API（使用 cast 显式类型转换）
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=cast(Iterable[ChatCompletionMessageParam], api_messages),
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,  # 从配置读取，不硬编码
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                extra_body=self._build_extra_body(),
                **kwargs,
            )

            # 解析响应
            choice = response.choices[0]
            usage = response.usage

            # 处理响应内容
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
                choice.message, "reasoning", None
            )

            logger.debug(
                "OpenAI响应: content=%s, reasoning_content=%s, finish_reason=%s",
                choice.message.content,
                reasoning,
                choice.finish_reason,
            )
            #  添加总 token 数
            logger.info(
                f"Token usage | prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens}, "
                f"total: {usage.prompt_tokens + usage.completion_tokens}"
            )

            llm_usage = LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )
            await self._record_usage(llm_usage)

            return LLMResponse(
                content=content or "",
                model=response.model,
                usage=llm_usage,
                finish_reason=choice.finish_reason or "stop",
            )

        except APITimeoutError as e:
            logger.error(
                "❌ OpenAI调用超时 - 模型: %s, base_url: %s, timeout: %s, 错误: %s",
                self.config.model,
                self.config.base_url,
                self.config.timeout,
                e,
            )
            raise LLMTimeoutError(f"OpenAI调用超时: {e}") from e
        except RateLimitError as e:
            logger.error(
                "❌ OpenAI速率限制 - 模型: %s, 错误: %s",
                self.config.model,
                e,
            )
            raise LLMRateLimitError(f"OpenAI速率限制: {e}") from e
        except (APIError, APIConnectionError) as e:
            logger.error(
                "❌ OpenAI调用失败 - 模型: %s, base_url: %s, 错误: %s",
                self.config.model,
                self.config.base_url,
                e,
                exc_info=True,
            )
            raise LLMError(f"OpenAI调用失败: {e}") from e
        except Exception as e:
            logger.error("未知错误: %s", e, exc_info=True)
            raise LLMError(f"OpenAI调用失败: {e}") from e

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """
        OpenAI流式聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            include_reasoning: 是否返回推理内容（reasoning_content）
            **kwargs: 其他参数

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）

        Raises:
            LLMError: 调用失败
        """
        try:
            # 记录使用的模型信息（添加max_tokens）
            logger.info(
                "🤖 调用流式LLM - 模型: %s, base_url: %s, temperature: %.2f, max_tokens: %s, timeout: %s, enable_think: %s",
                self.config.model,
                self.config.base_url,
                temperature or self.config.temperature,
                max_tokens or self.config.max_tokens or "未设置",
                self.config.timeout,
                self.config.enable_thinking,
            )

            # 打印输入消息（调试用）
            for i, msg in enumerate(messages):
                content_preview = msg.content[:5000] if len(msg.content) > 5000 else msg.content
                logger.info(f"📝 消息[{i}] role={msg.role.value}: {content_preview}")

            # 准备消息
            api_messages = self._prepare_messages(messages)

            # 调用流式API（使用 cast 显式类型转换）
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=cast(Iterable[ChatCompletionMessageParam], api_messages),
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stream=True,
                extra_body=self._build_extra_body(),
                **kwargs,
            )

            # 逐个生成内容片段
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content if delta.content else None
                    reasoning = None

                    # 如果需要推理内容，尝试获取reasoning_content
                    if include_reasoning:
                        reasoning = getattr(delta, "reasoning_content", None) or getattr(
                            delta, "reasoning", None
                        )

                    # 只在有内容或推理时yield
                    if content or reasoning:
                        yield (content or "", reasoning)

        except APITimeoutError as e:
            logger.error("OpenAI流式调用超时: %s", e)
            raise LLMTimeoutError(f"OpenAI流式调用超时: {e}") from e
        except (APIError, APIConnectionError) as e:
            logger.error("OpenAI流式调用失败: %s", e, exc_info=True)
            raise LLMError(f"OpenAI流式调用失败: {e}") from e
        except Exception as e:
            logger.error("未知错误: %s", e, exc_info=True)
            raise LLMError(f"OpenAI流式调用失败: {e}") from e

    async def close(self) -> None:
        """关闭OpenAI客户端，释放HTTP连接"""
        try:
            await self.client.close()
            logger.debug("OpenAI客户端已关闭")
        except Exception as e:
            logger.warning(f"关闭OpenAI客户端时出错: {e}")


async def create_openai_client(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> OpenAIClient:
    """
    创建OpenAI客户端（从环境变量读取默认值）

    Args:
        api_key: API密钥
        model: 模型名称（可选，默认从环境变量读取）
        base_url: 基础URL（可选，默认从环境变量读取）
        temperature: 温度参数（可选，默认从环境变量读取）
        max_tokens: 最大输出token数（可选，默认从环境变量读取）
        timeout: 超时时间（秒）（可选，默认从环境变量读取）
        max_retries: 最大重试次数（可选，默认从环境变量读取）

    Returns:
        OpenAI客户端实例
    """

    settings = get_settings()

    config = ModelConfig(
        provider=LLMProvider.OPENAI,
        model=model or settings.llm_model,
        api_key=api_key,
        base_url=base_url or settings.llm_base_url,
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        top_p=settings.llm_top_p,
        frequency_penalty=settings.llm_frequency_penalty,
        presence_penalty=settings.llm_presence_penalty,
        timeout=timeout or settings.llm_timeout,
        max_retries=max_retries or settings.llm_max_retries,
        top_k=settings.llm_top_k,
        min_p=settings.llm_min_p,
        repetition_penalty=settings.llm_repetition_penalty,
        enable_thinking=settings.llm_enable_think,
    )

    return OpenAIClient(config)
