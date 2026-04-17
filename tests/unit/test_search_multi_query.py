"""
Unit tests for multi-query expansion in hybrid_retrieve.

These tests verify that expansion_queries and extra_vectors are actually used
during retrieval and improve recall for queries that would otherwise miss
relevant chunks.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.deps import Store, reset_store, set_store
from app.retrieval.bm25 import BM25Index
from app.retrieval.search import hybrid_retrieve
from tests.fakes.mistral import FakeMistralClient


@pytest.fixture
def two_chunk_store():
    """Store with two orthogonal chunks: A and B."""
    # Chunk A: row 0, vector [1,0,0,0]
    # Chunk B: row 1, vector [0,1,0,0]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    bm25 = BM25Index()
    bm25.add(0, "alpha bravo")
    bm25.add(1, "charlie delta")
    bm25.finalize()
    state = Store(embeddings=vectors, bm25=bm25, ready_rows={0, 1})
    set_store(state)
    yield state
    reset_store()


def test_multi_query_improves_recall(two_chunk_store):
    """Expansion query should surface chunk B even when the base query only matches A."""
    fake = FakeMistralClient(dim=4)
    # Base query is collinear with chunk A
    fake.register_vector("query_a", np.array([1, 0, 0, 0], dtype=np.float32))
    # Expansion query is collinear with chunk B
    fake.register_vector("query_b", np.array([0, 1, 0, 0], dtype=np.float32))

    candidates = hybrid_retrieve(
        fake,
        query="query_a",
        mask=None,
        active_rows=two_chunk_store.ready_rows,
        expansion_queries=["query_b"],
        per_query_k=5,
    )

    rows = {c.row for c in candidates}
    assert 0 in rows, "Chunk A must appear (matches base query)"
    assert 1 in rows, "Chunk B must appear (matches expansion query)"


def test_no_expansion_misses_unrelated_chunk(two_chunk_store):
    """Without expansion, a query collinear with A should not rank B at the top."""
    fake = FakeMistralClient(dim=4)
    fake.register_vector("query_a", np.array([1, 0, 0, 0], dtype=np.float32))

    candidates = hybrid_retrieve(
        fake,
        query="query_a",
        mask=None,
        active_rows=two_chunk_store.ready_rows,
        per_query_k=5,
    )

    # Chunk A should be top
    assert candidates[0].row == 0


def test_extra_vector_surfaces_chunk(two_chunk_store):
    """extra_vectors (e.g. HyDE embedding) should make chunk B retrievable."""
    fake = FakeMistralClient(dim=4)
    # Base query maps to chunk A
    fake.register_vector("query_a", np.array([1, 0, 0, 0], dtype=np.float32))

    hyde_vec = np.array([0, 1, 0, 0], dtype=np.float32)  # collinear with chunk B

    candidates = hybrid_retrieve(
        fake,
        query="query_a",
        mask=None,
        active_rows=two_chunk_store.ready_rows,
        extra_vectors=[hyde_vec],
        per_query_k=5,
    )

    rows = {c.row for c in candidates}
    assert 1 in rows, "Chunk B must appear via extra_vectors"


def test_expansion_empty_list_behaves_like_baseline(two_chunk_store):
    """Passing expansion_queries=[] must produce the same result as omitting it."""
    fake = FakeMistralClient(dim=4)
    fake.register_vector("query_a", np.array([1, 0, 0, 0], dtype=np.float32))

    baseline = hybrid_retrieve(
        fake,
        query="query_a",
        mask=None,
        active_rows=two_chunk_store.ready_rows,
        per_query_k=5,
    )
    with_empty = hybrid_retrieve(
        fake,
        query="query_a",
        mask=None,
        active_rows=two_chunk_store.ready_rows,
        expansion_queries=[],
        per_query_k=5,
    )

    assert [c.row for c in baseline] == [c.row for c in with_empty]
