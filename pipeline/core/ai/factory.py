"""
LLM客户端工厂

根据配置创建相应的LLM客户端，支持场景化配置
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from pipeline.core.ai.base import BaseLLMClient, LLMRetryClient
from pipeline.core.ai.embedding_provider import (
    EmbeddingClientProvider,
    ResolvedEmbeddingConfig,
)
from pipeline.core.ai.llm import OpenAIClient
from pipeline.core.ai.models import LLMProvider, ModelConfig
from pipeline.core.config import Settings, get_settings
from pipeline.exceptions import ConfigError
from pipeline.utils import get_logger

if TYPE_CHECKING:
    from pipeline.core.ai.embedding import EmbeddingClient

logger = get_logger("ai.factory")


async def _load_db_config(type: str = "llm", scenario: str = "general") -> dict[str, Any] | None:
    """
    从数据库加载模型配置（通用函数）- 已禁用，默认使用环境变量

    降级策略（针对 LLM）：
    1. 查询 type + scenario 的专用配置
    2. 降级到 type + 'general'
    3. 返回 None（使用环境变量兜底）

    对于 Embedding 等：
    - 直接查 type + scenario（通常是 general）

    Args:
        type: 模型类型 (llm/embedding)
        scenario: 使用场景

    Returns:
        配置字典或None
    """
    # 默认使用环境变量配置，不再从数据库加载
    logger.debug(f"使用环境变量配置: type={type}, scenario={scenario}")
    return None


def _build_judge_config(settings: Settings) -> dict[str, Any]:
    """Build LLM config dict for the judge scenario from Settings.

    Reads JUDGE_LLM_* env vars. Falls back to main LLM only if
    judge_allow_fallback is explicitly enabled. Defaults:
      - temperature=0.0
      - enable_thinking=False
    """
    judge_key = settings.judge_llm_api_key
    judge_model = settings.judge_llm_model

    if not judge_key or not judge_model:
        if settings.judge_allow_fallback:
            logger.warning("Judge LLM 配置不完整，fallback 到主 LLM（judge_allow_fallback=True）")
            return {
                "model": settings.llm_model,
                "api_key": settings.llm_api_key,
                "base_url": settings.llm_base_url,
                "temperature": 0.0,
                "max_tokens": settings.judge_llm_max_tokens,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": settings.judge_llm_timeout,
                "max_retries": settings.judge_llm_max_retries,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "enable_thinking": settings.judge_llm_enable_thinking,
            }
        missing = []
        if not judge_key:
            missing.append("JUDGE_LLM_API_KEY")
        if not judge_model:
            missing.append("JUDGE_LLM_MODEL")
        raise ConfigError(
            f"Judge LLM 配置不完整，缺少: {', '.join(missing)}。"
            f"请设置对应环境变量，或启用 judge_allow_fallback=true 回退到主 LLM。"
        )

    return {
        "model": judge_model,
        "api_key": judge_key,
        "base_url": settings.judge_llm_base_url,
        "temperature": 0.0,
        "max_tokens": settings.judge_llm_max_tokens,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": settings.judge_llm_timeout,
        "max_retries": settings.judge_llm_max_retries,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "enable_thinking": settings.judge_llm_enable_thinking,
    }


async def create_llm_client(
    scenario: str = "general",
    model_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseLLMClient | LLMRetryClient:
    """
    创建LLM客户端（统一入口，支持场景化配置）

    配置优先级（从高到低）：
    1. model_config 显式传入
    2. 环境变量配置 (兜底)

    Args:
        scenario: 场景标识，默认 'general'
            - 'extract' : 事项提取
            - 'search'  : 搜索
            - 'chat'    : 对话
            - 'summary' : 摘要
            - 'system'  : 系统（Agent创建等）
            - 'general' : 通用（默认）
            - 'judge'   : LLM Judge 评测（独立配置，temperature=0.0，thinking 默认关闭）

        model_config: LLM配置字典（可选）
            {
                'model': 'gpt-4',
                'api_key': 'sk-xxx',
                'base_url': 'http://localhost:32768/v1',
                'temperature': 0.7,
                'max_tokens': 8000,
                ...
            }
            - 如果传入：直接使用（最高优先级）
            - 如果不传：自动从配置管理器获取

        **kwargs: 零散参数（向后兼容）

    Returns:
        LLM客户端实例

    Raises:
        ConfigError: 无法获取有效配置时抛出

    Examples:
        # 方式1：只传场景，自动获取配置（推荐）
        >>> client = await create_llm_client(scenario='extract')

        # 方式2：显式传入配置
        >>> client = await create_llm_client(
        ...     scenario='extract',
        ...     model_config={'model': 'gpt-4', 'temperature': 0.1}
        ... )

        # 方式3：使用默认通用场景
        >>> client = await create_llm_client()

    说明：
    - 统一使用 OpenAIClient（兼容 OpenAI 官方 + 302.AI 中转）
    - 通过 base_url 区分不同服务商
    """
    settings = get_settings()

    # ============ 配置合并（三层优先级）============

    # Layer 3: 环境变量兜底（场景化）
    if scenario == "judge":
        config = _build_judge_config(settings)
    else:
        config = {
            "model": settings.llm_model,
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_base_url,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "top_p": settings.llm_top_p,
            "frequency_penalty": settings.llm_frequency_penalty,
            "presence_penalty": settings.llm_presence_penalty,
            "timeout": settings.llm_timeout,
            "max_retries": settings.llm_max_retries,
            "top_k": settings.llm_top_k,
            "min_p": settings.llm_min_p,
            "repetition_penalty": settings.llm_repetition_penalty,
            "enable_thinking": settings.llm_enable_think,
        }

    # Layer 2: 数据库配置已移除，直接使用环境变量

    # Layer 1: 显式配置（最高优先级）
    if model_config:
        config.update(model_config)
        logger.debug(f"🎯 使用显式配置: scenario={scenario}")

    # 兼容零散参数（向后兼容）
    if kwargs:
        config.update(kwargs)

    # ============ 验证必需参数 ============
    if not config.get("api_key"):
        raise ConfigError(
            f"❌ LLM配置错误：缺少 API Key！\n" f"场景: {scenario}\n" f"请检查环境变量 LLM_API_KEY"
        )

    if not config.get("model"):
        raise ConfigError(f"❌ LLM配置错误：缺少模型名称！场景: {scenario}")

    # ============ 构建配置对象 ============
    model_config_obj = ModelConfig(
        provider=LLMProvider.OPENAI,  # 统一使用 OPENAI（兼容所有中转服务）
        model=config["model"],
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        frequency_penalty=config["frequency_penalty"],
        presence_penalty=config["presence_penalty"],
        timeout=config["timeout"],
        max_retries=config["max_retries"],
        top_k=config["top_k"],
        min_p=config["min_p"],
        repetition_penalty=config["repetition_penalty"],
        enable_thinking=config.get("enable_thinking", False),
    )

    # ============ 创建客户端（统一使用OpenAIClient）============
    # OpenAIClient 兼容：OpenAI 官方 + 302.AI 中转 + 其他兼容服务
    base_client = OpenAIClient(model_config_obj)

    # 包装重试机制
    with_retry = config.get("with_retry", True)
    if with_retry:
        logger.debug(
            f"✅ 创建LLM客户端（带重试）: scenario={scenario}",
            extra={
                "scenario": scenario,
                "model": config["model"],
                "base_url": config.get("base_url") or "OpenAI官方",
                "max_retries": config["max_retries"],
            },
        )
        return LLMRetryClient(base_client)

    logger.debug(
        f"✅ 创建LLM客户端: scenario={scenario}",
        extra={
            "scenario": scenario,
            "model": config["model"],
        },
    )
    return base_client


# LLM clients are caller-owned. Embedding uses one provider-owned shared client.


async def resolve_embedding_config(
    scenario: str = "general",
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedEmbeddingConfig:
    """Resolve and validate embedding configuration in one place."""
    settings = get_settings()
    config: dict[str, Any] = {
        "model": settings.embedding_model_name,
        "api_key": settings.embedding_api_key or settings.llm_api_key,
        "base_url": settings.embedding_base_url or settings.llm_base_url,
        "dimensions": settings.embedding_request_dimensions,
        "timeout": 60.0,
        "max_retries": 3,
    }

    if settings.use_db_config:
        db_config = await _load_db_config(type="embedding", scenario=scenario)
        if db_config:
            extra_data = db_config.get("extra_data") or {}
            if "dimensions" in extra_data:
                db_config["dimensions"] = extra_data["dimensions"]
            config.update(db_config)

    if overrides:
        config.update(overrides)

    if not config.get("api_key"):
        raise ConfigError(
            f"Embedding configuration is missing an API key for scenario '{scenario}'. "
            "Set EMBEDDING_API_KEY or LLM_API_KEY."
        )
    if not config.get("model"):
        raise ConfigError(f"Embedding configuration is missing a model for scenario '{scenario}'.")

    return ResolvedEmbeddingConfig(
        model=str(config["model"]),
        base_url=str(config["base_url"]) if config.get("base_url") else None,
        api_key=str(config["api_key"]),
        dimensions=(int(config["dimensions"]) if config.get("dimensions") is not None else None),
        timeout=float(config.get("timeout", 60.0)),
        max_retries=int(config.get("max_retries", 3)),
    )


def _build_embedding_client(config: ResolvedEmbeddingConfig) -> "EmbeddingClient":
    from pipeline.core.ai.embedding import EmbeddingClient

    return EmbeddingClient(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        dimensions=config.dimensions,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


_embedding_provider = EmbeddingClientProvider(
    config_resolver=resolve_embedding_config,
    client_factory=_build_embedding_client,
)


async def create_embedding_client(
    scenario: str = "general",
    embedding_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "EmbeddingClient":
    """Create a caller-owned embedding client.

    The caller must close this client. Use ``get_embedding_client`` for the
    provider-owned shared client.
    """
    overrides = dict(embedding_config or {})
    overrides.update(kwargs)
    client = await _embedding_provider.create_owned(scenario, overrides or None)
    return cast("EmbeddingClient", client)


async def get_embedding_client(scenario: str = "general") -> "EmbeddingClient":
    """Return the provider-owned shared embedding client."""
    client = await _embedding_provider.get(scenario)
    return cast("EmbeddingClient", client)


async def reset_embedding_client() -> None:
    """Close and clear the provider-owned shared embedding client."""
    await _embedding_provider.reset()


async def close_all_clients() -> None:
    """Close provider-owned AI clients."""
    await _embedding_provider.aclose()
