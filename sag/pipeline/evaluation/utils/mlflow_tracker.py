"""
MLflow 跟踪器模块

提供 MLflow 实验跟踪的封装类，支持配置化初始化、指标记录和自动资源管理。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


def get_local_ip():
    """返回本机标识，用于 MLflow run/experiment 命名。

    改用 hostname 而非连接 8.8.8.8 探测出网 IP：去掉对公网地址的硬编码依赖，
    且 hostname 比 IP 更可读。函数名沿用历史命名，实际返回 hostname。
    """
    import socket

    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


@dataclass
class MLflowConfig:
    """MLflow 配置

    Attributes:
        uri: MLflow tracking URI，None 表示使用默认
        experiment: 实验名称
        dataset_name: 数据集名称
        bench_size: 基准大小，可选
        enable_qa: 是否启用 QA
        vector_top_k: 向量检索 top-k
        max_entities: 最大实体数
        max_hops: 最大跳数
        max_results: 最大结果数
        description: 实验描述（包含参数 JSON），可选
    """

    uri: str | None = None
    experiment: str = "default"
    dataset_name: str = ""
    bench_size: int | None = None
    enable_qa: bool = False
    vector_top_k: int = 50
    max_entities: int = 50
    max_hops: int = 3
    max_results: int = 10
    description: str | None = None


class MLflowTracker:
    """MLflow 跟踪器封装类

    封装 MLflow 的初始化、参数记录、指标记录和运行结束等操作。
    支持上下文管理器模式，自动管理 MLflow run 的生命周期。

    Example:
        >>> config = MLflowConfig(uri="http://localhost:5000", experiment="my_exp")
        >>> with MLflowTracker(config, questions) as tracker:
        ...     tracker.log_evaluation_metrics(10, 5, 0, 10, step=0)
        ...     tracker.log_recall_metrics({"recall@1": 0.8, "recall@5": 0.9}, step=0)
    """

    def __init__(self, config: MLflowConfig, questions: list[Any]):
        """初始化 MLflow 跟踪器

        Args:
            config: MLflow 配置对象
            questions: 问题列表，用于记录问题总数
        """
        self.config = config
        self.questions = questions
        self._run = None
        self._logger = None
        self._run_id = None

    @property
    def run(self):
        """获取 MLflow run 对象"""
        return self._run

    @property
    def active(self) -> bool:
        """检查是否有活跃的 MLflow run"""
        return self._run is not None

    def start(self):
        """启动 MLflow 运行

        设置 tracking URI，创建/获取实验，启动 run 并记录参数。

        Returns:
            self: 支持链式调用

        Raises:
            Exception: MLflow 初始化失败时抛出异常
        """
        import mlflow

        if self._logger is None:
            from .logging_utils import get_logger

            self._logger = get_logger(__name__)

        # 设置 tracking URI
        if self.config.uri:
            mlflow.set_tracking_uri(self.config.uri)

        # 创建或获取实验
        try:
            mlflow.set_experiment(self.config.experiment)
        except Exception as e:
            if "deleted experiment" in str(e):
                # 如果实验被删除，创建新实验
                new_exp_name = (
                    f"{self.config.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                mlflow.set_experiment(new_exp_name)
                self._logger.warning(f"原实验已删除，创建新实验: {new_exp_name}")
            else:
                raise

        # 创建运行 - 添加 description 参数
        run_name = f"{get_local_ip()}_retrieval_{self.config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run = mlflow.start_run(run_name=run_name, description=self.config.description)
        self._run_id = self._run.info.run_id

        # 记录参数
        self._log_params()

        self._logger.info(f"MLflow 监控已启用 (实验: {self.config.experiment}, 运行: {run_name})")
        return self

    def _log_params(self):
        """记录初始化参数到 MLflow"""
        import mlflow

        mlflow.log_params(
            {
                "dataset": self.config.dataset_name,
                "bench_size": self.config.bench_size or 0,
                "total_questions": len(self.questions),
                "enable_qa": self.config.enable_qa,
                "vector_top_k": self.config.vector_top_k,
                "max_entities": self.config.max_entities,
                "max_hops": self.config.max_hops,
                "max_results": self.config.max_results,
            }
        )

    def _ensure_active_run(self) -> bool:
        """确保 MLflow run 处于 active 状态。

        当 run 在服务端被删除或不再活跃时，自动禁用后续日志写入。
        """
        if not self._run_id:
            return False
        import mlflow

        try:
            active = mlflow.active_run()
            if not active or active.info.run_id != self._run_id:
                # 重新绑定到当前 run（如果还存在）
                mlflow.start_run(run_id=self._run_id)
            return True
        except Exception as e:
            if self._logger:
                self._logger.warning(f"MLflow run 无法恢复为 active，后续将跳过日志: {e}")
            self._run = None
            self._run_id = None
            return False

    def _safe_log(self, log_fn, *args, **kwargs):
        """安全记录 MLflow 指标，失败则降级为跳过。"""
        if not self._ensure_active_run():
            return
        try:
            log_fn(*args, **kwargs)
        except Exception as e:
            if self._logger:
                self._logger.warning(f"MLflow 记录失败，后续将跳过日志: {e}")
            # 防止持续报错
            self._run = None
            self._run_id = None
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

    def log_batch_metrics(self, metrics: dict[str, float], step: int):
        """记录批次指标

        Args:
            metrics: 指标字典，key 为指标名，value 为指标值
            step: 当前步骤/批次号
        """
        import mlflow

        self._safe_log(mlflow.log_metrics, metrics, step=step)

    def log_metric(self, key: str, value: float, step: int):
        """记录单个指标

        Args:
            key: 指标名称
            value: 指标值
            step: 当前步骤/批次号
        """
        import mlflow

        self._safe_log(mlflow.log_metric, key, value, step=step)

    def log_recall_metrics(self, pooled_recall: dict[str, float], step: int):
        """记录检索指标（Recall/Precision/F1 @K）

        自动从指标名中提取 K 值并记录。

        Args:
            pooled_recall: 指标字典，格式如 {"Recall@1": 0.8, "Precision@5": 0.6, "F1@5": 0.7}
            step: 当前步骤/批次号
        """
        import mlflow

        for metric_name, score in pooled_recall.items():
            if "@" not in metric_name:
                continue
            prefix, k_str = metric_name.split("@", 1)
            try:
                k = int(k_str)
            except ValueError:
                continue
            prefix_norm = prefix.strip().lower()
            if prefix_norm not in {"recall", "precision", "f1"}:
                continue
            self._safe_log(mlflow.log_metric, f"{prefix_norm}_at_{k}", score, step=step)

    def _log_prefixed(self, metrics: dict[str, float], prefix: str, step: int):
        """给每个 metric key 加统一域前缀后上报（跳过非数值）。

        Args:
            metrics: 指标字典（key 已含维度后缀如 _cum/_batch，不含域前缀）
            prefix: 域前缀，如 "time_" / "token_"
            step: 当前步骤/批次号
        """
        import mlflow

        for metric_name, value in metrics.items():
            if not isinstance(value, int | float):
                continue
            self._safe_log(mlflow.log_metric, f"{prefix}{metric_name}", float(value), step=step)

    def log_timing_metrics(self, timings_avg: dict[str, float], step: int):
        """记录累计的各 Step 平均耗时，加 ``time_`` 域前缀。

        与 log_recall_metrics 同步每 bench 上报（step=batch_index）。MLflow 时间
        只保留 ``*_cum`` 口径，不上报 ``*_batch`` 或 ``*_percall``。

        Args:
            timings_avg: benchmark 侧算好的耗时字典（key 已含 _cum/_batch 维度后缀）
            step: 当前步骤/批次号
        """
        self._log_prefixed(timings_avg, "time_", step)

    def log_token_metrics(self, tokens_avg: dict[str, float], step: int):
        """记录 Step7 LLM token 平均（输入/输出，每问题 & 每次调用），加 ``token_`` 域前缀。

        与 log_timing_metrics 对称，每 bench 上报。指标名如
        token_step7_prompt_cum / token_step7_completion_batch / token_step7_prompt_percall_cum。

        Args:
            tokens_avg: benchmark 侧算好的 token 字典（key 已含 _cum/_batch 维度后缀）
            step: 当前步骤/批次号
        """
        self._log_prefixed(tokens_avg, "token_", step)

    def log_supplementary_metrics(self, metrics: dict[str, float], step: int):
        """Record an already ordered set of versioned search diagnostics.

        Callers provide complete metric names.  The ``supp_01`` through
        ``supp_04`` prefixes are the UI ordering contract: cumulative
        with-retry time, cumulative no-retry time, cumulative token statistics,
        then batch token statistics.  Core Recall/Precision/F1 metrics must be
        logged before this method is called.
        """

        import mlflow

        numeric_metrics = {
            metric_name: float(value)
            for metric_name, value in metrics.items()
            if isinstance(value, int | float)
        }
        if numeric_metrics:
            self._safe_log(mlflow.log_metrics, numeric_metrics, step=step)

    def log_evaluation_metrics(
        self, full_count: int, partial_count: int, zero_count: int, current_idx: int, step: int
    ):
        """记录评估指标

        记录完整的召回、部分召回和零召回的统计数量。

        Args:
            full_count: 完全召回的问题数
            partial_count: 部分召回的问题数
            zero_count: 零召回的问题数
            current_idx: 当前处理的问题索引
            step: 当前步骤/批次号
        """
        import mlflow

        self._safe_log(
            mlflow.log_metrics,
            {
                "full_recall_count": full_count,
                "partial_recall_count": partial_count,
                "zero_recall_count": zero_count,
                "questions_processed": current_idx,
            },
            step=step,
        )

    def end(self):
        """结束 MLflow 运行

        安全地结束当前 MLflow run，允许多次调用（幂等）。
        """
        if self._run:
            try:
                import mlflow

                mlflow.end_run()
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"MLflow 结束运行失败: {e}")
            self._run = None
            self._run_id = None
            if self._logger:
                self._logger.info("MLflow 运行已结束")

    def __enter__(self):
        """上下文管理器入口"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动结束运行"""
        self.end()
        return False


__all__ = [
    "MLflowTracker",
    "MLflowConfig",
    "get_local_ip",
]
