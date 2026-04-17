"""
Unit tests for the vector storage and persistence layer.

These tests verify the atomic saving, loading, and concatenation of embedding
matrices stored as NumPy files.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.storage.vector_store import concat_and_save, load_matrix, save_matrix_atomic


def test_save_load_roundtrip(tmp_path):
    """
    Verify that a matrix can be saved to and loaded from disk while preserving its values and shape.
    """
    path = tmp_path / "e.npy"
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    save_matrix_atomic(path, matrix)
    loaded = load_matrix(path, expected_dim=3)

    assert np.allclose(loaded, matrix)


def test_load_missing_returns_empty(tmp_path):
    """
    Verify that attempting to load a non-existent matrix returns a correctly shaped empty matrix
    rather than raising an error.
    """
    path = tmp_path / "e.npy"

    matrix = load_matrix(path, expected_dim=4)

    # Should have 0 rows but preserve the embedding dimension
    assert matrix.shape == (0, 4)


def test_concat_and_save_atomic(tmp_path):
    """
    Verify the ability to atomically append new vectors to an existing matrix on disk.
    """
    path = tmp_path / "e.npy"
    # Start with a 2x3 matrix of zeros
    save_matrix_atomic(path, np.zeros((2, 3), dtype=np.float32))
    # Add a 3x3 matrix of ones
    delta = np.ones((3, 3), dtype=np.float32)

    # concat_and_save should return the new combined matrix and save it to disk
    new_matrix = concat_and_save(path, np.zeros((2, 3), dtype=np.float32), delta)
    loaded = load_matrix(path, expected_dim=3)

    # Final shape should be (2+3)x3 = 5x3
    assert new_matrix.shape == (5, 3)
    assert loaded.shape == (5, 3)


def test_dim_mismatch_raises(tmp_path):
    """
    Verify that loading a matrix with an unexpected embedding dimension raises a ValueError.
    """
    path = tmp_path / "e.npy"
    # Save with dimension 4
    save_matrix_atomic(path, np.zeros((2, 4), dtype=np.float32))

    # Try to load expecting dimension 3
    with pytest.raises(ValueError, match="dim"):
        load_matrix(path, expected_dim=3)
