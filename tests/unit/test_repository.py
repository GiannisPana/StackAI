"""
Unit tests for the database repository layer (CRUD operations).

These tests verify the high-level functions used to manage document and chunk
records in the SQLite database.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk
from app.storage.db import get_connection, init_schema, transaction
from app.storage.repository import (
    DocumentRow,
    get_document_by_sha256,
    insert_chunks,
    insert_document,
    update_chunk_embedding_rows,
    update_document_status,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """
    Fixture to set up a clean, temporary database for repository tests.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")

    import app.config

    app.config._settings = None
    init_schema()


def _chunk(ord_: int) -> Chunk:
    """
    Helper function to create a synthetic Chunk object for testing.
    """
    return Chunk(
        page=1,
        ordinal=ord_,
        text=f"text {ord_}",
        token_count=2,
        bbox=(0, 0, 10, 10),
        source="pdf_text",
    )


def test_insert_document_roundtrip(db):
    """
    Verify that a document can be inserted and then retrieved by its SHA256 hash.
    """
    conn = get_connection()
    try:
        with transaction(conn):
            doc_id = insert_document(conn, filename="a.pdf", sha256="abc", num_pages=2, num_chunks=3)

        # Retrieve the document
        row = get_document_by_sha256(conn, "abc")

        assert row is not None
        assert row.id == doc_id
        assert row.status == "processing"  # Initial status should be 'processing'
        assert row.is_deleted == 0
    finally:
        conn.close()


def test_insert_chunks_and_update_embedding_rows(db):
    """
    Verify the workflow of inserting chunks, updating their row mapping in the
    vector store, and updating document status.
    """
    conn = get_connection()
    try:
        # Step 1: Insert document and chunks
        with transaction(conn):
            doc_id = insert_document(conn, filename="a.pdf", sha256="abc", num_pages=1, num_chunks=2)
            insert_chunks(conn, doc_id, [_chunk(0), _chunk(1)])

        # Initially, embedding_row should be None
        rows = conn.execute(
            "SELECT embedding_row FROM chunks WHERE doc_id=? ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
        assert [row["embedding_row"] for row in rows] == [None, None]

        # Step 2: Update embedding row mapping and mark as ready
        with transaction(conn):
            # Map chunks to start at row 50 in the vector matrix
            update_chunk_embedding_rows(conn, doc_id, base_row=50)
            update_document_status(conn, doc_id, "ready")

        # Verify row mapping
        rows = conn.execute(
            "SELECT embedding_row FROM chunks WHERE doc_id=? ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
        assert [row["embedding_row"] for row in rows] == [50, 51]

        # Verify document status
        doc = get_document_by_sha256(conn, "abc")
        assert doc is not None
        assert doc.status == "ready"
    finally:
        conn.close()
