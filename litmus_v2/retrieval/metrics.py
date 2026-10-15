"""
Phase 6: Information Retrieval Metrics.
Defines Hit@K, MRR, nDCG@K, and MAP with strict standard mathematical definitions.
Zero fallback tricks. Zero invented denominators.
"""

import math
from typing import Dict, List, Sequence, Set, Union


def hit_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Hit@K (binary):
        Returns 1.0 if at least one relevant document appears in top-k, else 0.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if bool(top_k & relevant_ids) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 0,
) -> float:
    """
    Reciprocal Rank (RR):
        1 / rank of the first relevant document (1-indexed).
        If no relevant document is found (or within k if k > 0), returns 0.0.
    """
    if not relevant_ids:
        return 0.0
    items = retrieved_ids[:k] if k > 0 else retrieved_ids
    for rank, doc_id in enumerate(items, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K (nDCG@K) for binary relevance:
        DCG@K = sum_{i=1}^k (rel_i / log2(i + 1))
        IDCG@K = sum_{i=1}^{min(k, |relevant|)} (1 / log2(i + 1))
        nDCG@K = DCG@K / IDCG@K

    Edge cases:
        - If |relevant| == 0: returns 0.0.
        - If IDCG@K == 0: returns 0.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    # DCG@K
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG@K
    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0

    return min(1.0, dcg / idcg)


def average_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Average Precision at K (AP@K):
        AP@K = (1 / min(k, |relevant|)) * sum_{i=1}^k [ P(i) * rel_i ]
        where P(i) is precision at cut-off i.

    Standard TREC definition:
        Denominator is |relevant| (or min(k, |relevant|) when truncated at K).
        We use min(k, |relevant|) so that a perfect top-K ranking achieves AP=1.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    denominator = min(k, len(relevant_ids))
    if denominator == 0:
        return 0.0

    return min(1.0, sum_precisions / denominator)


def compute_retrieval_metrics(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[Set[str]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Computes mean Hit@K, MRR, nDCG@K, and MAP across a dataset of queries.
    """
    n_queries = len(all_retrieved)
    if n_queries != len(all_relevant):
        raise ValueError(f"Query count mismatch: {n_queries} vs {len(all_relevant)}")
    if n_queries == 0:
        return {}

    results: Dict[str, float] = {}

    # MRR (over all returned or max K)
    mrr = sum(reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)) / n_queries
    results["mrr"] = round(mrr, 4)

    for k in k_values:
        hit = sum(hit_at_k(ret, rel, k=k) for ret, rel in zip(all_retrieved, all_relevant)) / n_queries
        ndcg = sum(ndcg_at_k(ret, rel, k=k) for ret, rel in zip(all_retrieved, all_relevant)) / n_queries
        map_k = sum(average_precision_at_k(ret, rel, k=k) for ret, rel in zip(all_retrieved, all_relevant)) / n_queries

        results[f"hit_at_{k}"] = round(hit, 4)
        results[f"ndcg_at_{k}"] = round(ndcg, 4)
        results[f"map_at_{k}"] = round(map_k, 4)

    return results
