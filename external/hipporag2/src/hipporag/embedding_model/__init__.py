import os

from .base import EmbeddingConfig, BaseEmbeddingModel
from .OpenAIEmbedding import OpenAIEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2",
                               global_config=None):
    # 如果配置了远端 embedding 服务地址，无论模型名是什么都走 API 客户端
    if global_config is not None and getattr(global_config, 'embedding_base_url', None):
        return OpenAIEmbeddingModel
    raise ValueError(
        "SAG-Benchmark external 集成仅支持远端 OpenAI 兼容 embedding 服务；"
        "本地模型（Contriever/GritLM/NV-Embed-v2）未随 external 包分发。"
    )
