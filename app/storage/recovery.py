"""Recovery logic for maintaining consistency between the DB and disk storage.

This module provides the 'run_recovery' function, which is called during
application startup to reconcile document metadata in SQLite with the
persistent embedding matrix (NumPy) and BM25 index (JSON).
"""

from __future__ import annotations

import numpy as np

from app.config import get_settings
from app.deps import get_store
from app.retrieval.bm25 import TOKENIZER_VERSION, BM25Index
from app.storage.bm25_store import load_bm25, save_bm25
from app.storage.db import get_connection, transaction
from app.storage.repository import (
    fetch_ready_chunks_for_rebuild,
    mark_processing_and_failed_as_failed,
    max_ready_row,
    ready_row_set,
)
from app.storage.vector_store import load_matrix, save_matrix_atomic


def run_recovery() -> None:
    """Reconciles the database state with the disk-based search structures.

    This function performs the following steps:
    1. Resets documents in 'processing' state to 'failed' (orphaned during crash).
    2. Truncates the embedding matrix to match the latest fully-published row in the DB.
    3. Rebuilds the BM25 index if documents were failed or if the index file is missing.
    4. Initializes the global 'Store' with reconciled data.

    This ensures that searches only return results for documents that were
    completely and successfully ingested.
    """
    settings = get_settings()
    store = get_store()

    conn = get_connection()
    try:
        with transaction(conn):
            # Cleanup documents that were interrupted during ingestion
            touched = mark_processing_and_failed_as_failed(conn)

        # Truncate the embedding matrix to discard orphaned rows from crashed ingestions
        expected_len = max_ready_row(conn) + 1
        matrix = load_matrix(settings.embeddings_path, expected_dim=settings.embedding_dim)
        
        if matrix.shape[0] > expected_len:
            # Corruption: Matrix is longer than DB records. Truncate it.
            matrix = matrix[:expected_len].copy()
            save_matrix_atomic(settings.embeddings_path, matrix)
        elif matrix.shape[0] < expected_len:
            # Corruption: Matrix is shorter than DB records. 
            # Roll back documents that reference missing rows to 'failed'.
            missing_threshold = matrix.shape[0]
            with transaction(conn):
                affected = conn.execute(
                    "SELECT DISTINCT doc_id FROM chunks WHERE embedding_row >= ?",
                    (missing_threshold,)
                ).fetchall()
                for r in affected:
                    did = r["doc_id"]
                    conn.execute("UPDATE chunks SET embedding_row = NULL WHERE doc_id = ?", (did,))
                    conn.execute("UPDATE documents SET status = 'failed' WHERE id = ?", (did,))
            # Re-calculate expected_len after rollback
            expected_len = max_ready_row(conn) + 1
            touched.extend([r["doc_id"] for r in affected])

        # Rebuild BM25 index from scratch if any documents were lost or file is missing
        need_rebuild = bool(touched) or not settings.bm25_path.exists()
        if not need_rebuild:
            bm25 = load_bm25(settings.bm25_path)
            if bm25.tokenizer_version != TOKENIZER_VERSION:
                need_rebuild = True
        if need_rebuild:
            bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
            for row, text in fetch_ready_chunks_for_rebuild(conn):
                bm25.add(row, text)
            bm25.finalize()
            save_bm25(settings.bm25_path, bm25)

        # Handle edge case of an empty system
        if matrix.size == 0:
            matrix = np.zeros((0, settings.embedding_dim), dtype=np.float32)

        # Update the global in-memory store
        store.embeddings = matrix
        store.bm25 = bm25
        store.ready_rows = ready_row_set(conn)

        # Verify the invariant: Matrix length must match the database's record
        assert store.embeddings.shape[0] == expected_len, (
            f"recovery invariant: matrix={store.embeddings.shape[0]} expected={expected_len}"
        )
    finally:
        conn.close()
