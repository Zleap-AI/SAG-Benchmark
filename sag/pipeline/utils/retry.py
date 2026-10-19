"""
重试工具模块

提供异常分类纯函数（is_retryable_error 等），供加载/批量写路径
（pipeline/modules/load/、pipeline/utils/batch.py）判断错误是否可重试。
LLM 侧的重试判定由 pipeline.core.ai.base 的 LLMRetryClient 自行实现，不依赖本模块。
"""

from sqlalchemy.exc import IntegrityError, OperationalError

from pipeline.exceptions import (
    RetryableError,
)


def is_retryable_db_error(error: Exception) -> bool:
    """
    判断数据库错误是否可重试

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    if isinstance(error, OperationalError):
        error_str = str(error)
        # 死锁和锁等待超时可重试
        if "1213" in error_str or "Deadlock" in error_str:
            return True
        if "1205" in error_str or "Lock wait timeout" in error_str:
            return True
        # 连接丢失可重试
        if "2013" in error_str or "Lost connection" in error_str:
            return True
        # 连接超时可重试
        if "2003" in error_str or "Can't connect" in error_str:
            return True
        # 其他 OperationalError 不可重试（如语法错误）
        return False

    # IntegrityError（唯一键冲突）不可重试
    if isinstance(error, IntegrityError):
        return False

    return False


def is_retryable_network_error(error: Exception) -> bool:
    """
    判断网络错误是否可重试

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    error_str = str(error).lower()

    # 连接超时、读取超时可重试
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # 连接被拒绝、连接重置可重试
    if "connection refused" in error_str or "connection reset" in error_str:
        return True

    # 临时性网络错误可重试
    if "temporary failure" in error_str or "network unreachable" in error_str:
        return True

    return False


def is_retryable_error(error: Exception) -> bool:
    """
    判断异常是否可重试（统一入口）

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    # 自定义可重试异常
    if isinstance(error, RetryableError):
        return True

    # 数据库错误
    if is_retryable_db_error(error):
        return True

    # 网络错误
    if is_retryable_network_error(error):
        return True

    return False
