"""
Unit tests for vector indexing and similarity search.

These tests verify the core numerical operations used for vector retrieval,
including normalization and top-K similarity search using inner products.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.vector_index import l2_normalize, top_k


def test_l2_normalize_unit_norm():
    """
    Verify that the L2 normalization function correctly produces unit vectors.
    """
    vector = np.array([3, 4], dtype=np.float32)

    normalized = l2_normalize(vector)

    # Resulting norm should be 1.0
    assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6


def test_top_k_sorted_desc():
    """
    Verify that the top_k search function returns results sorted by descending similarity score.
    """
    # Create 3 vectors and normalize them
    matrix = np.array([[1, 0], [0.9, 0.43589], [0, 1]], dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Query is identical to the first vector
    query = np.array([1, 0], dtype=np.float32)

    result = top_k(matrix, query, k=3, mask=None)

    # Result should be sorted by index [0, 1, 2] since they are in order of similarity to [1, 0]
    assert [row for row, _ in result] == [0, 1, 2]
    # Scores should be non-increasing
    assert result[0][1] >= result[1][1] >= result[2][1]


def test_top_k_mask_filters_rows():
    """
    Verify that the 'mask' parameter correctly restricts the search to specific document indices.
    """
    matrix = np.eye(4, dtype=np.float32)
    query = matrix[0]

    # Mask only allows indices 1 and 2
    result = top_k(matrix, query, k=4, mask={1, 2})

    # Index 0 (the exact match) should be filtered out
    assert {row for row, _ in result} == {1, 2}


def test_top_k_empty_matrix_returns_empty():
    """
    Verify that searching an empty index returns an empty list of results.
    """
    matrix = np.zeros((0, 4), dtype=np.float32)
    query = np.array([1, 0, 0, 0], dtype=np.float32)

    assert top_k(matrix, query, k=5, mask=None) == []
