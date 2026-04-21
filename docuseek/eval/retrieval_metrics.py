"""
docuseek/eval/retrieval_metrics.py
------------------------------------
Pure functions for retrieval evaluation metrics.

All functions take retrieved_urls (ordered list, most relevant first)
and gold_urls (list of ground-truth relevant URLs) and return a float
in [0, 1]. Binary relevance is assumed throughout: a URL is either
relevant (1) or not (0).
"""

import math


def recall_at_k(retrieved_urls: list[str], gold_urls: list[str], k: int) -> float:
    """
    Fraction of gold URLs found in the top-k retrieved results.

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.
        k:              Cutoff rank.

    Returns:
        Float in [0, 1]. 1.0 means all gold URLs appear in top-k.
    """
    if not gold_urls:
        return 0.0
    gold_set = set(gold_urls)
    return len(gold_set.intersection(retrieved_urls[:k])) / len(gold_set)


def precision_at_k(retrieved_urls: list[str], gold_urls: list[str], k: int) -> float:
    """
    Fraction of top-k retrieved URLs that are relevant.

    The denominator is always k — a retriever that returns fewer than k
    results is penalised, which is what we want when comparing systems.

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.
        k:              Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if not gold_urls or k == 0:
        return 0.0
    gold_set = set(gold_urls)
    return len(gold_set.intersection(retrieved_urls[:k])) / k


def reciprocal_rank(retrieved_urls: list[str], gold_urls: list[str]) -> float:
    """
    Reciprocal of the rank of the first relevant URL.

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.

    Returns:
        Float in [0, 1]. 0.0 if no relevant URL is found.
    """
    gold_set = set(gold_urls)
    for i, url in enumerate(retrieved_urls, start=1):
        if url in gold_set:
            return 1 / i
    return 0.0


def average_precision_at_k(retrieved_urls: list[str], gold_urls: list[str], k: int) -> float:
    """
    Average Precision at rank k.

    Sums Precision@i at each rank i (within top-k) where a relevant URL
    appears, and divides by the total number of relevant URLs capped at k.

    This is the standard MAP definition — dividing by the number of
    relevant docs rather than the number found ensures the metric
    penalises missing relevant URLs.

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.
        k:              Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if not gold_urls or k == 0:
        return 0.0
    gold_set = set(gold_urls)

    hits = 0
    precision_sum = 0.0
    for i, url in enumerate(retrieved_urls[:k], start=1):
        if url in gold_set:
            hits += 1
            precision_sum += hits / i

    denominator = min(len(gold_set), k)
    return precision_sum / denominator


def ndcg_at_k(retrieved_urls: list[str], gold_urls: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at rank k.

    Rewards finding relevant documents AND ranking them higher.
    Binary relevance: gain=1 for relevant URLs, 0 otherwise.

    DCG@K  = sum over top-k of gain / log2(rank + 1), rank is 1-indexed
    IDCG@K = DCG of the ideal ranking (all gold URLs at top ranks, capped at k)
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.
        k:              Cutoff rank.

    Returns:
        Float in [0, 1]. 1.0 means all relevant URLs are ranked at the top.
    """
    if not gold_urls or k == 0:
        return 0.0
    gold_set = set(gold_urls)

    dcg = sum(
        1 / math.log2(i + 1)  # gain is 1 or 0
        for i, url in enumerate(retrieved_urls[:k], start=1)
        if url in gold_set
    )

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_all(
    retrieved_urls: list[str],
    gold_urls: list[str],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single question.

    Args:
        retrieved_urls: Ordered list of retrieved URLs, best first.
        gold_urls:      Ground-truth relevant URLs.
        k:              Cutoff rank applied to all @k metrics.

    Returns:
        Dict with keys: recall@k, precision@k, rr, ap@k, ndcg@k.
        To get MRR and MAP@k, average rr and ap@k across all questions.
    """
    return {
        f"recall@{k}": recall_at_k(retrieved_urls, gold_urls, k),
        f"precision@{k}": precision_at_k(retrieved_urls, gold_urls, k),
        "rr": reciprocal_rank(retrieved_urls, gold_urls),
        f"ap@{k}": average_precision_at_k(retrieved_urls, gold_urls, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_urls, gold_urls, k),
    }
