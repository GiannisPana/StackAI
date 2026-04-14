"""Maximal Marginal Relevance (MMR) for search result diversification.

MMR aims to reduce redundancy in search results by selecting items that are
both relevant to the query and diverse compared to already selected items.
"""

from __future__ import annotations

import numpy as np


def mmr_select(*, vectors: np.ndarray, relevance: list[float], k: int, lambda_: float) -> list[int]:
    """Select k items using the Maximal Marginal Relevance algorithm.

    MMR = argmax_{D_i \in R\S} [λ * Sim1(D_i, Q) - (1-λ) * max_{D_j \in S} Sim2(D_i, D_j)]
    where S is the set of selected items and R is the set of candidate items.

    Args:
        vectors: Matrix of embeddings for candidate items (shape: (N, dim)).
        relevance: List of relevance scores (e.g., cosine similarity to query).
        k: Number of items to select.
        lambda_: Diversity parameter (1.0 = pure relevance, 0.0 = pure diversity).

    Returns:
        A list of indices for the selected items, in selection order.
    """
    n = vectors.shape[0]
    if n == 0:
        return []

    k = min(k, n)
    rel = np.asarray(relevance, dtype=np.float32)
    remaining = set(range(n))
    selected: list[int] = []

    # Step 1: Seed the selection with the single most relevant item
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    # Step 2: Greedily select k-1 more items based on the MMR score
    while len(selected) < k and remaining:
        # Compute similarities between all remaining candidates and already selected items
        selected_matrix = vectors[selected]
        candidates = sorted(remaining)
        candidate_matrix = vectors[candidates]
        # Sim2(D_i, D_j) for all candidates D_i and selected items D_j
        sims = candidate_matrix @ selected_matrix.T
        # max_{D_j \in S} Sim2(D_i, D_j)
        max_sims = sims.max(axis=1)
        # Relevance scores for the candidates
        candidate_rel = rel[candidates]
        # Calculate MMR score: λ * Relevance - (1-λ) * Max Similarity with Selected
        scores = lambda_ * candidate_rel - (1 - lambda_) * max_sims
        # Select the candidate with the highest MMR score
        best_local = int(np.argmax(scores))
        best = candidates[best_local]
        selected.append(best)
        remaining.remove(best)

    return selected
