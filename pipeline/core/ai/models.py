"""
LLM基础模型

定义LLM调用的消息、响应等数据模型
"""

from enum import Enum

from pydantic import BaseModel, Field

from pipeline.core.config.settings import DEFAULT_LLM_MAX_RETRIES


class LLMProvider(str, Enum):
    """LLM提供商"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMRole(str, Enum):
    """消息角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """LLM消息模型"""

    role: LLMRole = Field(..., description="角色")
    content: str = Field(..., description="消息内容")

    def to_dict(self) -> dict[str, str]:
        """转换为字典"""
        # Pylint 误认为 role 是 FieldInfo，实际上是 LLMRole 枚举
        return {"role": self.role.value, "content": self.content}  # pylint: disable=no-member


class LLMUsage(BaseModel):
    """Token使用统计"""

    prompt_tokens: int = Field(default=0, description="输入token数")
    completion_tokens: int = Field(default=0, description="输出token数")
    total_tokens: int = Field(default=0, description="总token数")


class LLMResponse(BaseModel):
    """LLM响应模型"""

    content: str = Field(..., description="响应内容")
    model: str = Field(..., description="使用的模型")
    usage: LLMUsage = Field(default_factory=LLMUsage, description="Token使用统计")
    finish_reason: str = Field(default="stop", description="结束原因")

    @property
    def total_tokens(self) -> int:
        """总token数"""
        # Pylint 误认为 usage 是 FieldInfo，实际上是 LLMUsage 模型
        return self.usage.total_tokens  # pylint: disable=no-member


class ModelConfig(BaseModel):
    """LLM配置（行为参数无默认值，仅由 factory 从 settings + DB + 显式 合并后注入）"""

    provider: LLMProvider = Field(..., description="提供商")
    model: str = Field(..., description="模型名称")
    api_key: str = Field(..., description="API密钥")
    base_url: str | None = Field(default=None, description="基础URL")

    # 行为参数（必填，由 factory 注入；默认值仅在 settings 中定义）
    temperature: float = Field(..., ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(..., ge=1, description="最大输出token数")
    top_p: float = Field(..., gt=0.0, le=1.0, description="top_p参数（vLLM要求>0）")
    frequency_penalty: float = Field(..., ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(..., ge=-2.0, le=2.0, description="存在惩罚")

    # 扩展采样参数（非 OpenAI 标准，经 extra_body 透传给 vLLM/SGLang 等后端）
    # top_k=-1 表示禁用；repetition_penalty=1.0 表示关闭
    top_k: int = Field(default=-1, ge=-1, description="top_k参数，-1表示禁用")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="min_p参数")
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="重复惩罚，1.0表示关闭")

    # 可靠性参数
    timeout: int = Field(default=600, ge=1, description="超时时间（秒）")
    max_retries: int = Field(
        default=DEFAULT_LLM_MAX_RETRIES,
        ge=0,
        description="最大重试次数",
    )

    # 思考模式（客户端级别，允许主 LLM 和 Judge LLM 独立控制）
    enable_thinking: bool = Field(default=False, description="思考模式开关")
