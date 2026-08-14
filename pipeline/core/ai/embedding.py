"""
Embedding生成服务

提供统一的文本向量化能力，所有模块共享
"""

from pipeline.core.config import get_settings
from pipeline.exceptions import AIError
from pipeline.utils import get_logger

logger = get_logger("ai.embedding")


class EmbeddingClient:
    """
    Embedding客户端

    统一的文本向量化服务，支持：
    - OpenAI Embedding API
    - 自定义Embedding服务
    - 本地Embedding模型（未来扩展）
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        初始化Embedding客户端

        Args:
            model: 模型名称（默认从配置读取）
            base_url: API地址（默认从配置读取）
            api_key: API密钥（默认从配置读取）
            dimensions: 透传给 embeddings.create 的 dimensions 参数（matryoshka 模型专用）。
                        None 表示不传该参数。注意 bge-large-en-v1.5 不支持。
        """
        from openai import AsyncOpenAI

        settings = get_settings()

        self.model = model or settings.embedding_model_name
        self.base_url = base_url or settings.embedding_base_url or settings.llm_base_url
        # ✅ 优先使用传入的 api_key，然后才是环境变量
        resolved_api_key = api_key or settings.embedding_api_key or settings.llm_api_key
        # 仅当显式指定时才透传 dimensions；bge-large-en-v1.5 等非 matryoshka 模型
        # 收到该参数会返回 400，因此默认 None 表示"不传"
        self.dimensions = dimensions

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._closed = False

        logger.info(
            "Embedding客户端初始化完成",
            extra={
                "model": self.model,
                "base_url": self.base_url or "default",
                "dimensions": self.dimensions or "server_default",
            },
        )

    def _create_kwargs(self) -> dict:
        """embeddings.create 的可选参数（dimensions 仅在显式配置时透传）"""
        return {"dimensions": self.dimensions} if self.dimensions else {}

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError("Embedding client is closed")

    async def generate(self, text: str) -> list[float]:
        """
        生成文本的embedding向量

        Args:
            text: 文本内容

        Returns:
            embedding向量

        Raises:
            AIError: 生成失败
        """
        self._ensure_open()
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
                **self._create_kwargs(),
            )
            embedding = response.data[0].embedding

            logger.debug(
                "生成embedding成功",
                extra={
                    "text_length": len(text),
                    "vector_dim": len(embedding),
                },
            )

            return embedding

        except Exception as e:
            logger.error(f"生成embedding失败: {e}", exc_info=True)
            raise AIError(f"生成embedding失败: {e}") from e

    async def batch_generate(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成embedding向量

        Args:
            texts: 文本列表

        Returns:
            embedding向量列表

        Raises:
            AIError: 生成失败
        """
        self._ensure_open()
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
                **self._create_kwargs(),
            )
            embeddings = [item.embedding for item in response.data]

            logger.debug(
                "批量生成embedding成功",
                extra={
                    "batch_size": len(texts),
                    "vector_dim": len(embeddings[0]) if embeddings else 0,
                },
            )

            return embeddings

        except Exception as e:
            logger.error(f"批量生成embedding失败: {e}", exc_info=True)
            raise AIError(f"批量生成embedding失败: {e}") from e

    async def probe_dimensions(self, probe_text: str = "dimension probe") -> int:
        """发一次最小 embedding 请求，返回服务端实际输出维度。

        不做缓存 —— 缓存由 pipeline.core.ai.embedding_dim 负责，
        避免把缓存策略和 HTTP 客户端耦合。
        """
        self._ensure_open()
        response = await self.client.embeddings.create(
            input=probe_text,
            model=self.model,
            **self._create_kwargs(),
        )
        dim = len(response.data[0].embedding)
        logger.info(
            "embedding 维度 probe 完成",
            extra={"model": self.model, "base_url": self.base_url or "default", "dim": dim},
        )
        return dim

    async def close(self) -> None:
        """Close the underlying HTTP client exactly once."""
        if self._closed:
            return
        self._closed = True
        await self.client.close()


async def generate_embedding(text: str) -> list[float]:
    """
    生成embedding的便捷函数

    Args:
        text: 文本内容

    Returns:
        embedding向量
    """
    from pipeline.core.ai.factory import get_embedding_client

    client = await get_embedding_client()
    return await client.generate(text)


async def batch_generate_embedding(texts: list[str]) -> list[list[float]]:
    """
    批量生成embedding的便捷函数

    Args:
        texts: 文本列表

    Returns:
        embedding向量列表
    """
    from pipeline.core.ai.factory import get_embedding_client

    client = await get_embedding_client()
    return await client.batch_generate(texts)
