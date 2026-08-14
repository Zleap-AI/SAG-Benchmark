"""
引擎数据模型

定义任务日志、阶段结果、任务结果等数据模型
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from pipeline.engine.config import OutputConfig
from pipeline.engine.enums import LogLevel, OutputMode, TaskStage, TaskStatus


class TaskLog(BaseModel):
    """任务日志"""

    timestamp: datetime
    stage: TaskStage
    level: LogLevel
    message: str
    extra: dict[str, Any] | None = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow()
        super().__init__(**data)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{time_str}] [{self.stage.value}] {self.level.value.upper()}: {self.message}"


class StageResult(BaseModel):
    """单个阶段的执行结果"""

    stage: TaskStage
    status: str  # success/failed/skipped
    data_ids: list[str] = Field(default_factory=list, description="数据ID列表")
    data_full: list[dict[str, Any]] = Field(default_factory=list, description="完整数据列表")
    stats: dict[str, Any] = Field(default_factory=dict, description="统计信息")
    error: str | None = None
    duration: float | None = None


class TaskResult(BaseModel):
    """任务执行结果"""

    model_config = {"arbitrary_types_allowed": True}

    # 任务基本信息
    task_id: str
    task_name: str
    status: TaskStatus

    # 数据标识
    source_config_id: str | None = None
    article_id: str | None = None

    # 各阶段结果
    load_result: StageResult | None = None
    extract_result: StageResult | None = None
    search_result: StageResult | None = None

    # 统计信息
    stats: dict[str, Any] = Field(default_factory=dict)

    # 日志
    logs: list[TaskLog] = Field(default_factory=list)
    error: str | None = None

    # 时间信息
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float | None = None

    def to_dict(self, output_config: OutputConfig) -> dict:
        """转换为字典"""
        data = {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "source_config_id": self.source_config_id,
            "article_id": self.article_id,
            "stats": self.stats,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
        }

        # 根据输出模式选择data_ids或data_full
        results_key = "data_ids" if output_config.mode == OutputMode.ID_ONLY else "data_full"

        # 添加各阶段结果
        if self.load_result:
            data["load"] = {
                "status": self.load_result.status,
                "results": getattr(self.load_result, results_key),
                "stats": self.load_result.stats,
            }

        if self.extract_result:
            data["extract"] = {
                "status": self.extract_result.status,
                "results": getattr(self.extract_result, results_key),
                "stats": self.extract_result.stats,
            }

        if self.search_result:
            data["search"] = {
                "status": self.search_result.status,
                "results": getattr(self.search_result, results_key),
                "stats": self.search_result.stats,
            }

        # 日志
        if output_config.include_logs:
            data["logs"] = [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "stage": log.stage.value,
                    "level": log.level.value,
                    "message": log.message,
                    "extra": log.extra,
                }
                for log in self.logs
            ]

        if self.error:
            data["error"] = self.error

        return data

    def to_json(self, output_config: OutputConfig) -> str:
        """转换为JSON"""
        import json

        data = self.to_dict(output_config)
        if output_config.pretty:
            return json.dumps(data, ensure_ascii=False, indent=2)
        return json.dumps(data, ensure_ascii=False)

    def is_success(self) -> bool:
        """任务是否成功"""
        return self.status == TaskStatus.COMPLETED
