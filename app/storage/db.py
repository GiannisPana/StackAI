"""SQLite database management for document metadata.

This module provides functions for database connectivity, transaction management,
and schema initialization. It ensures data consistency and optimal performance
for concurrent access.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Creates a new SQLite connection with optimized settings.

    It enables foreign keys and uses Write-Ahead Logging (WAL) mode to
    support concurrent readers and a single writer without blocking.
    """
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, isolation_level=None)
    # Ensure database-level referential integrity
    conn.execute("PRAGMA foreign_keys = ON")
    # Enable WAL mode for better concurrency performance
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Context manager for explicit transaction management."""
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_schema() -> None:
    """Initializes the database schema and verifies configuration consistency.

    If the database is being created or updated, it ensures that the
    'embedding_dim' in the database matches the current application configuration.
    This prevents mismatches that would cause runtime errors during retrieval.
    """
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        conn.executescript(ddl)
        chunk_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "section_title" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")
        existing = {row["key"] for row in conn.execute("SELECT key FROM meta")}
        seed = {
            "embedding_model": settings.embedding_model,
            "embedding_dim": str(settings.embedding_dim),
            "schema_version": "1",
        }
        for key, value in seed.items():
            if key not in existing:
                conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))
            elif key == "embedding_dim":
                # Guard against loading data with different vector dimensions
                current = conn.execute(
                    "SELECT value FROM meta WHERE key = 'embedding_dim'"
                ).fetchone()["value"]
                if int(current) != settings.embedding_dim:
                    raise RuntimeError(
                        f"embedding_dim mismatch: persisted={current}, configured={settings.embedding_dim}"
                    )
    finally:
        conn.close()
