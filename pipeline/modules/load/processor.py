"""
文档处理器

负责生成向量
"""

from pipeline.utils import estimate_tokens, get_logger

logger = get_logger("modules.load.processor")


class DocumentProcessor:
    """文档处理器，负责向量生成"""

    def __init__(
        self,
        llm_client=None,
        embedding_model_name: str | None = None,
    ) -> None:
        """
        初始化文档处理器

        Args:
            llm_client: 保留参数（兼容性），不使用
            embedding_model_name: 向量模型名称（如果不提供，从配置读取）
        """
        from pipeline.core.config import get_settings

        settings = get_settings()

        # 优先使用传入的模型名，否则从配置读取
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        logger.info(
            "文档处理器初始化完成",
            extra={
                "embedding_model_name": self.embedding_model_name,
            },
        )

    async def generate_embedding(self, text: str, instruction: str | None = None) -> list[float]:
        """
        生成文本向量

        Args:
            text: 文本内容
            instruction: query 侧非对称指令（纯文本），仅非对称模型的 query 使用；None=不传

        Returns:
            向量列表

        Raises:
            AIError: 向量生成失败
        """
        try:
            import time

            from pipeline.core.ai.factory import get_embedding_client
            from pipeline.exceptions import AIError
            from pipeline.utils import is_retryable_error

            total_start = time.perf_counter()

            truncate_start = time.perf_counter()
            truncated_text = self._truncate_content(text, max_tokens=8000)
            truncate_time = time.perf_counter() - truncate_start

            logger.debug(f"生成向量，文本长度: {len(text)}字符")

            api_start = time.perf_counter()
            embedding_client = await get_embedding_client(scenario="general")
            embedding = await embedding_client.generate(truncated_text, instruction=instruction)
            api_time = time.perf_counter() - api_start

            total_time = time.perf_counter() - total_start

            logger.info(
                f"向量生成耗时统计 - "
                f"总耗时: {total_time:.3f}s, "
                f"文本截断: {truncate_time:.3f}s ({truncate_time / total_time * 100:.1f}%), "
                f"API调用: {api_time:.3f}s ({api_time / total_time * 100:.1f}%), "
                f"向量维度: {len(embedding)}"
            )

            return embedding

        except Exception as e:
            from pipeline.exceptions import AIError
            from pipeline.utils import is_retryable_error

            if is_retryable_error(e):
                logger.warning(f"向量生成失败（可重试）: {e}")
                raise AIError(f"向量生成失败（临时性错误）: {e}") from e
            else:
                logger.error(f"向量生成失败（不可重试）: {e}")
                raise AIError(f"向量生成失败（永久性错误）: {e}") from e

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量（单次 API 调用）

        与 generate_embedding() 语义相同，但将多个文本合并为一次 API 请求，
        显著减少网络往返次数。返回顺序与输入顺序一致。

        Args:
            texts: 文本列表

        Returns:
            向量列表，顺序与输入一致

        Raises:
            AIError: 向量生成失败
        """
        if not texts:
            return []

        try:
            import time

            from pipeline.core.ai.factory import get_embedding_client
            from pipeline.exceptions import AIError
            from pipeline.utils import is_retryable_error

            total_start = time.perf_counter()

            truncated_texts = [self._truncate_content(t, max_tokens=8000) for t in texts]

            api_start = time.perf_counter()
            embedding_client = await get_embedding_client(scenario="general")
            embeddings = await embedding_client.batch_generate(truncated_texts)
            api_time = time.perf_counter() - api_start

            total_time = time.perf_counter() - total_start

            logger.info(
                f"批量向量生成耗时统计 - "
                f"批次大小: {len(texts)}, "
                f"总耗时: {total_time:.3f}s, "
                f"API调用: {api_time:.3f}s, "
                f"向量维度: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except Exception as e:
            from pipeline.exceptions import AIError
            from pipeline.utils import is_retryable_error

            if is_retryable_error(e):
                logger.warning(f"批量向量生成失败（可重试）: {e}")
                raise AIError(f"批量向量生成失败（临时性错误）: {e}") from e
            else:
                logger.error(f"批量向量生成失败（不可重试）: {e}")
                raise AIError(f"批量向量生成失败（永久性错误）: {e}") from e

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """
        截断内容以适应 token 限制

        Args:
            content: 原始内容
            max_tokens: 最大 token 数

        Returns:
            截断后的内容
        """
        estimated_tokens = estimate_tokens(content)

        if estimated_tokens <= max_tokens:
            return content

        # 按比例截断，留 10% 余量
        ratio = max_tokens / estimated_tokens
        target_length = int(len(content) * ratio * 0.9)
        truncated = content[:target_length]

        logger.debug(
            f"内容截断: {len(content)}字符 -> {len(truncated)}字符 "
            f"({estimated_tokens} tokens -> ~{max_tokens} tokens)"
        )

        return truncated
