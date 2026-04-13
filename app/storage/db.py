from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_schema() -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        conn.executescript(ddl)
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
                current = conn.execute(
                    "SELECT value FROM meta WHERE key = 'embedding_dim'"
                ).fetchone()["value"]
                if int(current) != settings.embedding_dim:
                    raise RuntimeError(
                        f"embedding_dim mismatch: persisted={current}, configured={settings.embedding_dim}"
                    )
    finally:
        conn.close()
