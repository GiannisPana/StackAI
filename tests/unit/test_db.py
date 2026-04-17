"""
Unit tests for the SQLite database integration and storage layer.

These tests verify schema initialization, transaction management, and
integrity constraints (like foreign keys) in the local SQLite database.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.storage.db import get_connection, init_schema, transaction


@pytest.fixture
def db(tmp_path, monkeypatch):
    """
    Fixture to set up a clean, temporary database for each test.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")

    import app.config

    # Reset internal settings state to ensure DATA_DIR is picked up
    app.config._settings = None
    init_schema()
    yield
    app.config._settings = None


def test_foreign_keys_enforced(db):
    """
    Verify that SQLite foreign key constraints are active and enforced.
    """
    conn = get_connection()
    try:
        # Attempt to insert a chunk with a non-existent doc_id
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    "INSERT INTO chunks (doc_id, ordinal, page, text, token_count, source) "
                    "VALUES (?,?,?,?,?,?)",
                    (9999, 0, 1, "x", 1, "pdf_text"),
                )
    finally:
        conn.close()


def test_transaction_rolls_back_on_error(db):
    """
    Verify that the transaction context manager correctly rolls back changes if an error occurs.
    """
    conn = get_connection()
    try:
        # Start a transaction, insert a document, then fail
        with pytest.raises(ValueError):
            with transaction(conn):
                conn.execute(
                    "INSERT INTO documents (filename, sha256, num_pages, num_chunks, status, ingested_at) "
                    "VALUES (?,?,?,?,?,?)",
                    ("a.pdf", "abc", 1, 0, "processing", "2026-04-13"),
                )
                raise ValueError("boom")

        # The document should not exist in the database after rollback
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 0
    finally:
        conn.close()


def test_meta_seeded(db):
    """
    Verify that the initial database schema includes correctly seeded metadata.
    """
    conn = get_connection()
    try:
        # The schema initialization should seed 'embedding_dim'
        dim = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
        assert dim is not None
        assert int(dim["value"]) == 1024
    finally:
        conn.close()
