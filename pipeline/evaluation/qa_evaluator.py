"""
QA 评估器

把 EM / F1 指标封装成一个面向「问题 → 预测答案 → gold 答案」场景的评估器，
对外风格与 pipeline/evaluation/evaluator.py 的 Evaluator 保持一致：
- evaluate_single_query: 单条样本
- evaluate: 批量宏平均

与检索评估（Evaluator，操作 retrieved/relevant 文档 id）正交：本类只关心
「预测答案字符串 vs gold 答案字符串」的语义匹配，用于 QA 阶段（检索回来的 chunk
送进 LLM 生成答案后）的打分。

使用示例：
    from pipeline.evaluation.qa_evaluator import QAEvaluator

    qa_eval = QAEvaluator()
    # 单条
    print(qa_eval.evaluate_single_query(
        predicted="Thomas C. Sudhof",
        gold=["Thomas C. Sudhof"],
    ))  # {"exact_match": 1.0, "f1": 1.0}

    # 批量
    print(qa_eval.evaluate(
        predictions=["June 1982", "Maradona"],
        gold_answers=[["June 1982"], ["Diego Maradona"]],
    ))  # {"exact_match": 0.5, "f1": 0.5}
"""

import numpy as np

from pipeline.utils import get_logger

from .metrics.qa_eval import QAExactMatch, QAF1Score

logger = get_logger("evaluation.qa_evaluator")


class QAEvaluator:
    """问答答案评估器（EM + F1）"""

    def __init__(self) -> None:
        self._em = QAExactMatch()
        self._f1 = QAF1Score()

    def evaluate_single_query(
        self,
        predicted: str,
        gold: list[str],
    ) -> dict[str, float]:
        """
        评估单条样本。

        Args:
            predicted: 预测答案字符串。
            gold: 该样本的 gold 答案列表（允许别名/多个等价答案）。

        Returns:
            {"exact_match": <0.0|1.0>, "f1": <0.0~1.0>}
        """
        em_pooled, _ = self._em.calculate_metric_scores(
            gold_answers=[gold], predicted_answers=[predicted], aggregation_fn=np.max
        )
        f1_pooled, _ = self._f1.calculate_metric_scores(
            gold_answers=[gold], predicted_answers=[predicted], aggregation_fn=np.max
        )
        return {"exact_match": em_pooled["ExactMatch"], "f1": f1_pooled["F1"]}

    def evaluate(
        self,
        predictions: list[str],
        gold_answers: list[list[str]],
    ) -> dict[str, float]:
        """
        批量评估（宏平均）。

        Args:
            predictions: 预测答案列表，与 gold_answers 逐项对齐。
            gold_answers: 每条样本的 gold 答案列表。

        Returns:
            {"exact_match": <平均>, "f1": <平均>}
        """
        if not predictions:
            logger.warning("预测结果为空，返回零分")
            return {"exact_match": 0.0, "f1": 0.0}

        em_pooled, _ = self._em.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predictions, aggregation_fn=np.max
        )
        f1_pooled, _ = self._f1.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predictions, aggregation_fn=np.max
        )
        return {
            "exact_match": round(float(em_pooled["ExactMatch"]), 4),
            "f1": round(float(f1_pooled["F1"]), 4),
        }

    def evaluate_per_example(
        self,
        predictions: list[str],
        gold_answers: list[list[str]],
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        """
        批量评估并返回逐条结果。

        Returns:
            (per_example, pooled):
                - per_example: [{"exact_match":, "f1":}, ...]
                - pooled: {"exact_match": <平均>, "f1": <平均>}
        """
        if not predictions:
            return [], {"exact_match": 0.0, "f1": 0.0}

        _, per_em = self._em.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predictions, aggregation_fn=np.max
        )
        _, per_f1 = self._f1.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predictions, aggregation_fn=np.max
        )
        per_example = [
            {"exact_match": e["ExactMatch"], "f1": f["F1"]} for e, f in zip(per_em, per_f1)
        ]
        return per_example, self.evaluate(predictions, gold_answers)


__all__ = ["QAEvaluator"]
