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


def test_init_schema_adds_missing_section_title_column(tmp_path, monkeypatch):
    """
    Existing databases created before section-aware chunking should be migrated
    in place so the new column appears without requiring manual intervention.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")

    import app.config

    app.config._settings = None
    db_path = tmp_path / "app.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO meta (key, value) VALUES ('embedding_dim', '1024');
            INSERT INTO meta (key, value) VALUES ('embedding_model', 'mistral-embed');
            INSERT INTO meta (key, value) VALUES ('schema_version', '1');

            CREATE TABLE documents (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                filename       TEXT NOT NULL,
                sha256         TEXT NOT NULL UNIQUE,
                num_pages      INTEGER NOT NULL,
                num_chunks     INTEGER NOT NULL,
                ocr_pages      INTEGER NOT NULL DEFAULT 0,
                status         TEXT NOT NULL CHECK(status IN ('processing','ready','failed')),
                is_deleted     INTEGER NOT NULL DEFAULT 0,
                deleted_at     TEXT,
                ingested_at    TEXT NOT NULL
            );

            CREATE TABLE chunks (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id         INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal        INTEGER NOT NULL,
                page           INTEGER NOT NULL,
                bbox_x0        REAL,
                bbox_y0        REAL,
                bbox_x1        REAL,
                bbox_y1        REAL,
                text           TEXT NOT NULL,
                token_count    INTEGER NOT NULL,
                embedding_row  INTEGER UNIQUE,
                source         TEXT NOT NULL CHECK(source IN ('pdf_text','ocr'))
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    init_schema()

    migrated = get_connection()
    try:
        columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(chunks)").fetchall()
        }
        assert "section_title" in columns
    finally:
        migrated.close()
