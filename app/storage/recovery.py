from __future__ import annotations

import numpy as np

from app.config import get_settings
from app.deps import get_store
from app.retrieval.bm25 import BM25Index
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
    settings = get_settings()
    store = get_store()

    conn = get_connection()
    try:
        with transaction(conn):
            touched = mark_processing_and_failed_as_failed(conn)

        expected_len = max_ready_row(conn) + 1
        matrix = load_matrix(settings.embeddings_path, expected_dim=settings.embedding_dim)
        if matrix.shape[0] > expected_len:
            matrix = matrix[:expected_len].copy()
            save_matrix_atomic(settings.embeddings_path, matrix)

        need_rebuild = bool(touched) or not settings.bm25_path.exists()
        if need_rebuild:
            bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
            for row, text in fetch_ready_chunks_for_rebuild(conn):
                bm25.add(row, text)
            bm25.finalize()
            save_bm25(settings.bm25_path, bm25)
        else:
            bm25 = load_bm25(settings.bm25_path)

        if matrix.size == 0:
            matrix = np.zeros((0, settings.embedding_dim), dtype=np.float32)

        store.embeddings = matrix
        store.bm25 = bm25
        store.ready_rows = ready_row_set(conn)

        assert store.embeddings.shape[0] == expected_len, (
            f"recovery invariant: matrix={store.embeddings.shape[0]} expected={expected_len}"
        )
    finally:
        conn.close()
