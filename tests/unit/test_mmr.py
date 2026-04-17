"""
Unit tests for the Maximal Marginal Relevance (MMR) re-ranking algorithm.

MMR is used to balance relevance and diversity in the final set of retrieved
documents, reducing redundancy in the context provided to the LLM.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.mmr import mmr_select


def test_lambda_1_pure_relevance():
    """
    Verify that with lambda=1.0, MMR behaves like pure relevance ranking.
    It should ignore diversity and just pick the top-N most relevant items.
    """
    # vector 0 and 1 are very similar, vector 2 is orthogonal to 0
    vectors = np.array([[1, 0, 0], [0.99, 0.14, 0], [0, 1, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = [0.9, 0.85, 0.8]

    # With lambda=1.0, it should pick index 0 and 1 (most relevant)
    chosen = mmr_select(vectors=vectors, relevance=relevance, k=2, lambda_=1.0)

    assert chosen == [0, 1]


def test_lambda_0_pure_diversity():
    """
    Verify that with lambda=0.0, MMR prioritizes diversity among top-ranked items.
    After picking the most relevant item, it should pick the one least similar to it.
    """
    vectors = np.array([[1, 0, 0], [0.99, 0.14, 0], [0, 1, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = [0.9, 0.85, 0.8]

    # With lambda=0.0, it should pick index 0 (most relevant) then 2 (most diverse from 0)
    chosen = mmr_select(vectors=vectors, relevance=relevance, k=2, lambda_=0.0)

    assert chosen[0] == 0
    assert chosen[1] == 2


def test_terminates_at_k():
    """
    Verify that MMR correctly stops after selecting exactly 'k' items.
    """
    vectors = np.eye(5, dtype=np.float32)
    relevance = [0.9, 0.8, 0.7, 0.6, 0.5]

    chosen = mmr_select(vectors=vectors, relevance=relevance, k=3, lambda_=0.5)

    assert len(chosen) == 3


def test_k_larger_than_pool():
    """
    Verify that MMR handles cases where 'k' is larger than the number of available items.
    It should return all available items in that case.
    """
    vectors = np.eye(2, dtype=np.float32)

    chosen = mmr_select(vectors=vectors, relevance=[1.0, 0.9], k=5, lambda_=0.5)

    assert len(chosen) == 2
