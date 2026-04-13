from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_matrix(path: Path, expected_dim: int) -> np.ndarray:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, matrix.astype(np.float32, copy=False))
    os.replace(tmp, path)


def concat_and_save(path: Path, current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if current.ndim != 2 or delta.ndim != 2 or current.shape[1] != delta.shape[1]:
        raise ValueError(f"dim mismatch: current={current.shape}, delta={delta.shape}")
    new_matrix = (
        np.concatenate([current, delta], axis=0)
        if current.size
        else delta.astype(np.float32, copy=False)
    )
    save_matrix_atomic(path, new_matrix)
    return new_matrix
