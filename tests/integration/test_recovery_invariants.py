"""
Integration tests for the storage recovery mechanism.

These tests verify that the 'run_recovery' process correctly synchronizes the
various storage backends (SQLite, Vector Store, BM25 Index). It ensures that
the SQLite metadata remains the "source of truth" and that external stores
are truncated or rebuilt to match its state.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.deps import Store, get_store, reset_store, set_store
from app.storage.db import get_connection, init_schema, transaction
from app.storage.recovery import run_recovery
from app.storage.repository import (
    insert_chunks,
    insert_document,
    max_ready_row,
    update_chunk_embedding_rows,
    update_document_status,
)
from app.storage.bm25_store import save_bm25
from app.storage.vector_store import load_matrix, save_matrix_atomic
from app.ingestion.chunker import Chunk
from app.retrieval.bm25 import BM25Index, tokenize


@pytest.fixture
def env(tmp_path, monkeypatch):
    """
    Setup a clean environment for recovery testing.
    """
    # Isolate data directory to a temporary path
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "4")

    import app.config

    # Reset settings and initialize fresh DB schema
    app.config._settings = None
    init_schema()
    
    yield
    
    # Teardown
    reset_store()
    app.config._settings = None


def test_recovery_on_empty_corpus(env):
    """
    Verify recovery behavior when the system is entirely empty.

    Ensures that an empty SQLite database results in an empty vector store and
    empty in-memory state after recovery.
    """
    set_store(Store(embeddings=np.zeros((0, 4), dtype=np.float32)))

    # Run the recovery logic
    run_recovery()

    # The store state should remain empty and consistent
    store = get_store()
    assert store.embeddings.shape == (0, 4)
    assert store.ready_rows == set()


def test_recovery_truncates_overlong_matrix(env):
    """
    Verify that an over-extended vector matrix is correctly truncated during recovery.

    Simulates a scenario where a crash occurred after writing more embedding
    rows than were successfully recorded/finalized in the SQLite database.
    """
    # 1. Manually save an "overlong" matrix file with 5 rows
    settings = get_settings()
    save_matrix_atomic(settings.embeddings_path, np.eye(5, 4).astype(np.float32))

    # 2. Record only 2 rows in SQLite (marking them as 'ready')
    conn = get_connection()
    try:
        with transaction(conn):
            # Create a mock document record
            doc_id = insert_document(conn, filename="a.pdf", sha256="a", num_pages=1, num_chunks=2)
            # Record 2 chunks for that document
            insert_chunks(
                conn,
                doc_id,
                [
                    Chunk(page=1, ordinal=0, text="alpha", token_count=1, bbox=(0, 0, 1, 1), source="pdf_text"),
                    Chunk(page=1, ordinal=1, text="beta", token_count=1, bbox=(0, 0, 1, 1), source="pdf_text"),
                ],
            )
            # Assign embedding row indices 0 and 1
            update_chunk_embedding_rows(conn, doc_id, base_row=0)
            # Finalize the document status
            update_document_status(conn, doc_id, "ready")
    finally:
        conn.close()

    # 3. Load the initial (overlong) state into memory
    set_store(Store(embeddings=load_matrix(settings.embeddings_path, expected_dim=4)))

    # 4. Run recovery to synchronize the systems
    run_recovery()

    # 5. Assert that the matrix was truncated to match the SQLite state (2 rows)
    store = get_store()
    assert max_ready_row(get_connection()) >= 1
    assert store.embeddings.shape[0] == 2
    assert store.ready_rows == {0, 1}
    
    # Assert that the BM25 index was also rebuilt to only include those 2 rows
    assert store.bm25 is not None
    for posts in store.bm25._postings.values():
        for row in posts.keys():
            assert row in {0, 1}


def test_recovery_rebuilds_bm25_when_tokenizer_version_changes(env):
    settings = get_settings()
    save_matrix_atomic(settings.embeddings_path, np.eye(1, 4).astype(np.float32))

    conn = get_connection()
    try:
        with transaction(conn):
            doc_id = insert_document(conn, filename="a.pdf", sha256="a", num_pages=1, num_chunks=1)
            insert_chunks(
                conn,
                doc_id,
                [
                    Chunk(
                        page=5,
                        ordinal=0,
                        text="foregoing provisions apply to all subcontracts",
                        token_count=6,
                        bbox=(0, 0, 1, 1),
                        source="pdf_text",
                        section_title="INSURANCE",
                    )
                ],
            )
            update_chunk_embedding_rows(conn, doc_id, base_row=0)
            update_document_status(conn, doc_id, "ready")
    finally:
        conn.close()

    stale = BM25Index()
    stale.add(0, "foregoing provisions apply to all subcontracts")
    stale.finalize()
    stale.tokenizer_version = "v2"
    save_bm25(settings.bm25_path, stale)

    set_store(Store(embeddings=load_matrix(settings.embeddings_path, expected_dim=4)))

    run_recovery()

    store = get_store()
    assert store.bm25 is not None
    assert store.bm25.tokenizer_version == "v3"
    hits = dict(store.bm25.top_k(tokenize("insurance subcontractor"), k=5))
    assert 0 in hits
