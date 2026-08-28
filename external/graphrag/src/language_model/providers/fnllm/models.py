# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

# [PATCH LAYER] 本文件替换上游 graphrag==2.0.0 包的同名文件。
# 修改内容：embedding 输入截断（_maybe_truncate_for_embedding）+ chat/embed 的 cost 埋点。
# 上游若将来提供等效扩展点，则删除本文件并恢复为上游版本。
# 同步机制：reproduce/_common.py ensure_overrides() 在每次运行时写入 .venv/site-packages。

"""A module containing fnllm model provider definitions."""

from collections.abc import AsyncGenerator

from fnllm.openai import (
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embeddings_llm,
)
from fnllm.openai.types.client import OpenAIChatLLM as FNLLMChatLLM
from fnllm.openai.types.client import OpenAIEmbeddingsLLM as FNLLMEmbeddingLLM

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.language_model_config import (
    LanguageModelConfig,
)
from graphrag.language_model.providers.fnllm.events import FNLLMEvents
from graphrag.language_model.providers.fnllm.utils import (
    _create_cache,
    _create_error_handler,
    _create_openai_config,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)


def _maybe_truncate_for_embedding(texts: list[str]) -> list[str]:
    """按环境变量 GRAPHRAG_EMBED_MAX_TOKENS 截断 embedding 输入（默认不截断）。

    用于 bge 这类有 512 输入上限、且服务端对超长直接返回 400 的服务。
    设 export GRAPHRAG_EMBED_MAX_TOKENS=300 后，每条输入按 cl100k_base 截断到 300 token
    （bge 实际约 <512），行为等价于 HippoRAG 的 embedding_max_seq_len 截断。
    """
    import os

    limit = os.getenv("GRAPHRAG_EMBED_MAX_TOKENS")
    if not limit:
        return texts
    try:
        from graphrag import cost_meter

        return cost_meter.truncate_texts(texts, int(limit))
    except Exception:  # noqa: BLE001  截断失败不影响主流程
        return texts


class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        # 成本计数：记录本次 chat 的 token 与调用（缓存命中不计）
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None)
            cost_meter.record_chat(
                input_tokens=getattr(_usage, "input_tokens", 0) if _usage else 0,
                output_tokens=getattr(_usage, "output_tokens", 0) if _usage else 0,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001  计数失败绝不影响主流程
            pass
        return BaseModelResponse(
            output=BaseModelOutput(content=response.output.content),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        text_list = _maybe_truncate_for_embedding(text_list)
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None) or getattr(
                response.output, "usage", None
            )
            _tok = getattr(_usage, "input_tokens", 0) if _usage else 0
            if not _tok:  # 服务端没返回 usage，用 tiktoken 兜底
                _tok = cost_meter.count_tokens(text_list)
            cost_meter.record_embed(
                input_tokens=_tok,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001
            pass
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        text = _maybe_truncate_for_embedding([text])[0]
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None) or getattr(
                response.output, "usage", None
            )
            _tok = getattr(_usage, "input_tokens", 0) if _usage else 0
            if not _tok:
                _tok = cost_meter.count_tokens([text])
            cost_meter.record_embed(
                input_tokens=_tok,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001
            pass
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        # 成本计数：记录本次 chat 的 token 与调用（缓存命中不计）
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None)
            cost_meter.record_chat(
                input_tokens=getattr(_usage, "input_tokens", 0) if _usage else 0,
                output_tokens=getattr(_usage, "output_tokens", 0) if _usage else 0,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001  计数失败绝不影响主流程
            pass
        return BaseModelResponse(
            output=BaseModelOutput(content=response.output.content),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        text_list = _maybe_truncate_for_embedding(text_list)
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None) or getattr(
                response.output, "usage", None
            )
            _tok = getattr(_usage, "input_tokens", 0) if _usage else 0
            if not _tok:  # 服务端没返回 usage，用 tiktoken 兜底
                _tok = cost_meter.count_tokens(text_list)
            cost_meter.record_embed(
                input_tokens=_tok,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001
            pass
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        text = _maybe_truncate_for_embedding([text])[0]
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        try:
            from graphrag import cost_meter

            _usage = getattr(response.metrics, "usage", None) or getattr(
                response.output, "usage", None
            )
            _tok = getattr(_usage, "input_tokens", 0) if _usage else 0
            if not _tok:
                _tok = cost_meter.count_tokens([text])
            cost_meter.record_embed(
                input_tokens=_tok,
                cache_hit=bool(getattr(response, "cache_hit", False)),
            )
        except Exception:  # noqa: BLE001
            pass
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))
