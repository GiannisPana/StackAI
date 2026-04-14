"""Vector similarity search and indexing utilities.

This module provides efficient NumPy-based operations for performing similarity
search (dot product) over an embedding matrix, with support for boolean masking.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit L2 norm.

    Args:
        vector: Input NumPy array.

    Returns:
        The L2-normalized vector. Returns the original vector if norm is zero.
    """
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
    """Compute dot-product scores and return the top-K indices and scores.

    If the matrix and query vector are L2-normalized, the dot product is
    equivalent to cosine similarity.

    Args:
        matrix: Embedding matrix of shape (num_docs, dim).
        query_vec: Query vector of shape (dim,).
        k: Number of top results to return.
        mask: Optional set of allowed row IDs to include in the search.

    Returns:
        A list of (row_index, score) tuples, sorted by score descending.
    """
    if matrix.shape[0] == 0:
        return []

    # Ensure query vector is float32 for consistent matrix multiplication
    query = query_vec.astype(np.float32)
    # Compute dot-product scores for all documents in a single operation
    scores = matrix @ query

    # Apply boolean mask if specified to exclude non-allowed rows
    if mask is not None:
        # Create a boolean array initialized to False
        allowed = np.zeros(matrix.shape[0], dtype=bool)
        # Flip bits to True for rows present in the mask set
        for row in mask:
            if 0 <= row < matrix.shape[0]:
                allowed[row] = True
        # Set scores of non-allowed rows to negative infinity to exclude them
        scores = np.where(allowed, scores, -np.inf)

    # Calculate number of documents with finite (allowed) scores
    finite_count = int(np.sum(np.isfinite(scores)))
    k = min(k, finite_count)
    if k <= 0:
        return []

    # Efficiently find indices of the top-K scores (not necessarily in order)
    # Using np.argpartition is O(N), much faster than a full sort O(N log N)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    # Sort only the selected top-K indices by their scores
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
