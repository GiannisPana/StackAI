"""
Unit tests for the Reciprocal Rank Fusion (RRF) algorithm.

RRF is used to combine multiple ranked lists (e.g., from vector search and
BM25 lexical search) into a single unified ranking.
"""

from __future__ import annotations

from app.retrieval.fusion import rrf


def test_rrf_classic_example():
    """
    Verify RRF with a classic example of two overlapping ranked lists.
    Items that appear high in both lists should be ranked highest in the fused output.
    """
    # List a: 1 (1st), 2 (2nd), 3 (3rd)
    a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    # List b: 3 (1st), 1 (2nd), 4 (3rd)
    b = [(3, 0.9), (1, 0.8), (4, 0.7)]

    fused = rrf([a, b], k=60)
    ids = [item_id for item_id, _ in fused]

    # IDs 1 and 3 should be at the top as they appear in both lists
    assert ids[0] in (1, 3)
    # ID 4 should still be included
    assert 4 in ids


def test_rrf_item_in_only_one_list_still_ranked():
    """
    Verify that items present in only one of the input lists are still included in the fused output.
    """
    fused = dict(rrf([[(1, 0.9)], [(2, 0.9)]], k=60))
    assert 1 in fused
    assert 2 in fused


def test_rrf_empty_lists_return_empty():
    """
    Verify that empty input lists or no input lists result in an empty fused list.
    """
    assert rrf([], k=60) == []
    assert rrf([[], []], k=60) == []


def test_rrf_constant_affects_relative_weight():
    """
    Verify that the 'k' constant in the RRF formula correctly affects the resulting scores.
    The formula is score = sum(1 / (k + rank)).
    """
    a = [(1, 0.0), (2, 0.0)]
    fused_k1 = dict(rrf([a], k=1))
    fused_k60 = dict(rrf([a], k=60))

    # A smaller k gives more weight to high ranks, resulting in higher absolute scores
    assert fused_k1[1] > fused_k60[1]
