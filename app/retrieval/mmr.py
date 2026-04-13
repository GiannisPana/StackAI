from __future__ import annotations

import numpy as np


def mmr_select(*, vectors: np.ndarray, relevance: list[float], k: int, lambda_: float) -> list[int]:
    n = vectors.shape[0]
    if n == 0:
        return []

    k = min(k, n)
    rel = np.asarray(relevance, dtype=np.float32)
    remaining = set(range(n))
    selected: list[int] = []

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        selected_matrix = vectors[selected]
        candidates = sorted(remaining)
        candidate_matrix = vectors[candidates]
        sims = candidate_matrix @ selected_matrix.T
        max_sims = sims.max(axis=1)
        candidate_rel = rel[candidates]
        scores = lambda_ * candidate_rel - (1 - lambda_) * max_sims
        best_local = int(np.argmax(scores))
        best = candidates[best_local]
        selected.append(best)
        remaining.remove(best)

    return selected
