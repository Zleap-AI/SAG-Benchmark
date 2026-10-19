"""
QA 评估指标

参考 HippoRAG 2 / MRQA 官方评测实现，提供：
- QAExactMatch: Exact Match (EM) 分数
- QAF1Score: Token 级 F1 分数

两者均基于 normalize_answer() 做归一化（小写、去冠词、去标点、合并空白），
当一条样本存在多个 gold answer 时用 aggregation_fn（默认 np.max）聚合。

使用示例：
    from pipeline.evaluation.metrics import QAExactMatch, QAF1Score

    em = QAExactMatch()
    f1 = QAF1Score()

    gold_answers = [["June 1982"], ["1982", "06/1982"]]
    predicted_answers = ["June 1982", "1982"]
    pooled_em, per_em = em.calculate_metric_scores(gold_answers, predicted_answers)
    pooled_f1, per_f1 = f1.calculate_metric_scores(gold_answers, predicted_answers)
"""

from collections import Counter
from collections.abc import Callable

import numpy as np

from pipeline.utils import get_logger

from ..utils.eval_utils import normalize_answer
from .base import BaseMetric

logger = get_logger("evaluation.metrics.qa_eval")


class QAExactMatch(BaseMetric):
    """Exact Match (EM) 指标 —— 归一化后预测答案与某个 gold 完全相等记 1.0，否则 0.0。"""

    metric_name: str = "qa_exact_match"

    def __init__(self) -> None:
        super().__init__()

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """
        计算 EM 分数。

        Args:
            gold_answers: 每个样本的 gold 答案列表（允许别名/多个等价答案）。
            predicted_answers: 每个样本的预测答案字符串。
            aggregation_fn: 跨多个 gold 的聚合函数（默认取最大）。

        Returns:
            (pooled, per_example):
                - pooled: {"ExactMatch": <样本平均>}
                - per_example: [{"ExactMatch": <0.0|1.0>}, ...]
        """
        assert len(gold_answers) == len(predicted_answers), (
            "Length of gold answers and predicted answers should be the same."
        )

        per_example: list[dict[str, float]] = []
        total = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            scores = [
                1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0
                for gold in gold_list
            ]
            aggregated = float(aggregation_fn(scores))
            per_example.append({"ExactMatch": aggregated})
            total += aggregated

        avg = total / len(gold_answers) if gold_answers else 0.0
        return {"ExactMatch": avg}, per_example


class QAF1Score(BaseMetric):
    """Token 级 F1 指标 —— 归一化后按 token 计算 precision/recall，再求调和平均。"""

    metric_name: str = "qa_f1_score"

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(predicted_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """
        计算 F1 分数。

        Args:
            gold_answers: 每个样本的 gold 答案列表。
            predicted_answers: 每个样本的预测答案字符串。
            aggregation_fn: 跨多个 gold 的聚合函数（默认取最大）。

        Returns:
            (pooled, per_example):
                - pooled: {"F1": <样本平均>}
                - per_example: [{"F1": <0.0~1.0>}, ...]
        """
        assert len(gold_answers) == len(predicted_answers), (
            "Length of gold answers and predicted answers should be the same."
        )

        per_example: list[dict[str, float]] = []
        total = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            scores = [self._compute_f1(gold, predicted) for gold in gold_list]
            aggregated = float(aggregation_fn(scores))
            per_example.append({"F1": aggregated})
            total += aggregated

        avg = total / len(gold_answers) if gold_answers else 0.0
        return {"F1": avg}, per_example


__all__ = ["QAExactMatch", "QAF1Score"]
