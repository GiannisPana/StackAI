"""Fusion strategies for combining multiple retrieval result sets.

This module implements Reciprocal Rank Fusion (RRF), a simple yet effective
technique for merging results from different retrieval systems (e.g., vector
search and keyword search) without needing to normalize scores.
"""

from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    RRF score is calculated as: score(d) = sum(1 / (k + rank(d)))
    where rank(d) is the 1-based index of document d in a given ranking.

    Args:
        rankings: A list of result sets, each containing (item_id, original_score)
            tuples. Only the order (rank) within each set matters, not the score.
        k: Smoothing constant to reduce the influence of items at very high ranks.

    Returns:
        A merged list of (item_id, rrf_score) tuples, sorted by score descending.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking):
            # The rank is 0-indexed here, so we add 1 to make it 1-based
            # for the standard RRF formula: 1 / (k + rank + 1)
            scores[item_id] += 1.0 / (k + rank + 1)
    # Sort documents by their accumulated RRF score in descending order
    return sorted(scores.items(), key=lambda item: -item[1])
