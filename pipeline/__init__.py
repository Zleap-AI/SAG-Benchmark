"""pipeline - 语义增强生成引擎

AI驱动的数据处理与聚合检索引擎
"""

__version__ = "0.1.0"
__author__ = "Zleap Team"
__email__ = "contact@zleap.ai"

from pipeline.engine import (
    DocumentLoadConfig,
    ExtractBaseConfig,
    LoadBaseConfig,
    LogLevel,
    OutputConfig,
    OutputMode,
    SearchBaseConfig,
    StageResult,
    TaskConfig,
    TaskLog,
    TaskResult,
    TaskStage,
    TaskStatus,
    PipelineEngine,
    pipelineEngine,
)
from pipeline.exceptions import LLMError, PipelineError, StorageError, ValidationError, pipelineError

__all__ = [
    # Version
    "__version__",
    # Engine
    "PipelineEngine",
    "pipelineEngine",
    "TaskConfig",
    "TaskResult",
    "TaskLog",
    "TaskStatus",
    "TaskStage",
    "StageResult",
    # Configs
    "LoadBaseConfig",
    "DocumentLoadConfig",
    "ExtractBaseConfig",
    "SearchBaseConfig",
    "OutputConfig",
    "OutputMode",
    "LogLevel",
    # Exceptions
    "PipelineError",
    "pipelineError",
    "LLMError",
    "StorageError",
    "ValidationError",
]
