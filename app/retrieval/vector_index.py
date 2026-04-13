from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def top_k(
    matrix: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    mask: set[int] | None,
) -> list[tuple[int, float]]:
    if matrix.shape[0] == 0:
        return []

    query = query_vec.astype(np.float32)
    scores = matrix @ query

    if mask is not None:
        allowed = np.zeros(matrix.shape[0], dtype=bool)
        for row in mask:
            if 0 <= row < matrix.shape[0]:
                allowed[row] = True
        scores = np.where(allowed, scores, -np.inf)

    finite_count = int(np.sum(np.isfinite(scores)))
    k = min(k, finite_count)
    if k <= 0:
        return []

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
