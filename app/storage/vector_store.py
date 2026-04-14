"""Persistent storage for document embedding matrices using NumPy.

This module provides functions for loading, saving, and updating large
embedding matrices. It employs an atomic 'stage-then-publish' pattern
to prevent data loss during file writes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_matrix(path: Path, expected_dim: int) -> np.ndarray:
    """Loads a NumPy matrix from disk and validates its dimensions.

    If the file does not exist, an empty matrix with the expected
    embedding dimension is returned.
    """
    if not path.exists():
        return np.zeros((0, expected_dim), dtype=np.float32)

    matrix = np.load(path)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError(
            f"embeddings dim mismatch: file has shape {matrix.shape}, expected (*, {expected_dim})"
        )
    return matrix


def save_matrix_atomic(path: Path, matrix: np.ndarray) -> None:
    """Saves a matrix to disk using an atomic rename operation.

    The matrix is first written to a temporary '.tmp' file. Once the write
    is complete and flushed to disk, the temporary file replaces the
    target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stage_matrix(path, matrix)
    # Atomic publish
    os.replace(tmp, path)


def stage_matrix(path: Path, matrix: np.ndarray) -> Path:
    """Writes a matrix to a temporary '.tmp' file.

    This is the first stage of the atomic save process.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, matrix.astype(np.float32, copy=False))
    return tmp


def build_concat_matrix(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Efficiently concatenates a new batch of embeddings to the existing matrix."""
    if current.ndim != 2 or delta.ndim != 2 or current.shape[1] != delta.shape[1]:
        raise ValueError(f"dim mismatch: current={current.shape}, delta={delta.shape}")
    return (
        np.concatenate([current, delta], axis=0)
        if current.size
        else delta.astype(np.float32, copy=False)
    )


def concat_and_save(path: Path, current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Concatenates new embeddings and saves the resulting matrix atomically.

    Returns:
        The newly created concatenated matrix.
    """
    new_matrix = build_concat_matrix(current, delta)
    save_matrix_atomic(path, new_matrix)
    return new_matrix
