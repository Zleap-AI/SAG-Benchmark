"""
pipeline 异常定义

所有自定义异常都继承自 PipelineError 基类
"""


class PipelineError(Exception):
    """pipeline 基础异常类"""

    def __init__(self, message: str, *args: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigError(PipelineError):
    """配置错误异常"""

    pass


class StorageError(PipelineError):
    """存储层异常"""

    pass


class DatabaseError(StorageError):
    """数据库异常"""

    pass


class CacheError(StorageError):
    """缓存异常"""

    pass


class LLMError(PipelineError):
    """LLM调用异常"""

    pass


class LLMTimeoutError(LLMError):
    """LLM调用超时异常"""

    pass


class LLMRateLimitError(LLMError):
    """LLM速率限制异常"""

    pass


class LLMRequestError(LLMError):
    """LLM请求不可恢复：参数、认证、权限或资源配置无效。"""

    pass


class LLMTransientError(LLMError):
    """LLM瞬态服务或网络错误，可由统一重试层重试。"""

    pass


class LLMResponseError(LLMError):
    """LLM响应不满足当前调用的输出契约，不应由调用方重复请求。"""

    pass


class AIError(PipelineError):
    """AI相关异常（包括LLM和Embedding）"""

    pass


class ValidationError(PipelineError):
    """数据验证异常"""

    pass


class LoadError(PipelineError):
    """文档加载异常"""

    pass


class EntityError(PipelineError):
    """实体处理异常"""

    pass


class ExtractError(PipelineError):
    """事项提取异常"""

    pass


class SearchError(PipelineError):
    """检索异常"""

    pass


class PromptError(PipelineError):
    """提示词异常"""

    pass


# ============ 可重试异常 ============


class RetryableError(PipelineError):
    """可重试异常基类（临时性错误，重试可能成功）"""

    pass


class NetworkError(RetryableError):
    """网络错误（连接超时、网络中断等）"""

    pass


class ResourceBusyError(RetryableError):
    """资源繁忙错误（数据库锁、并发冲突等）"""

    pass


class ServiceUnavailableError(RetryableError):
    """服务不可用错误（外部服务暂时不可用）"""

    pass


# ============ 不可重试异常 ============


class NonRetryableError(PipelineError):
    """不可重试异常基类（永久性错误，重试无意义）"""

    pass


class InvalidInputError(NonRetryableError):
    """无效输入错误（数据格式错误、参数非法等）"""

    pass


class ResourceNotFoundError(NonRetryableError):
    """资源不存在错误（文件不存在、记录不存在等）"""

    pass


class PermissionDeniedError(NonRetryableError):
    """权限错误（访问被拒绝、认证失败等）"""

    pass
