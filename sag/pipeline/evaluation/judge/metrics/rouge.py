"""ROUGE Score aligned with GraphRAG-Benchmark/Evaluation/metrics/rouge.py.

Deterministic metric; no LLM calls. Uses rouge-score library.
"""

from rouge_score import rouge_scorer


async def compute_rouge_score(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure",
) -> float:
    """Compute ROUGE score between generated answer and ground truth.

    Args:
        answer: Generated response text.
        ground_truth: Reference ground truth text.
        rouge_type: 'rouge1', 'rouge2', or 'rougeL'.
        mode: 'fmeasure', 'precision', or 'recall'.

    Returns:
        ROUGE score in [0.0, 1.0].
    """
    if not ground_truth.strip() or not answer.strip():
        return 0.0

    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)
