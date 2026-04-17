"""
Unit tests for basic hybrid search functionality.

These tests verify that the hybrid retrieval logic correctly combines vector
and BM25 search results, respecting filters and masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.deps import Store, reset_store, set_store
from app.retrieval.bm25 import BM25Index
from app.retrieval.search import hybrid_retrieve
from tests.fakes.mistral import FakeMistralClient


@pytest.fixture
def store():
    """
    Fixture to set up a mock Store with pre-populated embeddings and BM25 index.
    """
    # Vectors for 3 documents
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    
    # BM25 index with 3 documents
    bm25 = BM25Index()
    bm25.add(0, "parental leave sixteen weeks")
    bm25.add(1, "remote work policy hybrid schedule")
    bm25.add(2, "office supplies procurement")
    bm25.finalize()
    
    # Global state store
    state = Store(embeddings=vectors, bm25=bm25, ready_rows={0, 1, 2})
    set_store(state)
    yield state
    reset_store()


def test_hybrid_retrieve_returns_candidates(store):
    """
    Verify that hybrid_retrieve returns the expected document candidates for a query.
    """
    fake = FakeMistralClient(dim=4)
    # Map "parental leave" to vector 0
    fake.register_vector("parental leave", np.array([1, 0, 0, 0], dtype=np.float32))

    # Search for "parental leave"
    candidates = hybrid_retrieve(
        fake, query="parental leave", mask=None, active_rows=store.ready_rows, per_query_k=5
    )

    # Document 0 should be the top candidate
    ids = [candidate.row for candidate in candidates]
    assert 0 in ids
    assert candidates[0].row == 0


def test_mask_filters_rows(store):
    """
    Verify that the 'mask' parameter correctly filters out documents from the search results.
    """
    fake = FakeMistralClient(dim=4)
    fake.register_vector("anything", np.array([1, 0, 0, 0], dtype=np.float32))

    # Mask only allows documents 1 and 2
    candidates = hybrid_retrieve(
        fake, query="anything", mask={1, 2}, active_rows=store.ready_rows, per_query_k=5
    )

    # Results should only contain allowed IDs
    assert {candidate.row for candidate in candidates}.issubset({1, 2})
