"""Data access layer for document and chunk metadata in SQLite.

This module provides high-level functions for interacting with the database,
abstracting away the underlying SQL queries and ensuring consistent data mapping.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class DocumentRow:
    """Represents a single document record in the database."""
    id: int
    filename: str
    sha256: str
    num_pages: int
    num_chunks: int
    ocr_pages: int
    status: str
    is_deleted: int
    deleted_at: str | None
    ingested_at: str


def _now() -> str:
    """Helper to get the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _to_document_row(row: sqlite3.Row | None) -> DocumentRow | None:
    """Maps a SQLite Row to a DocumentRow dataclass."""
    if row is None:
        return None
    return DocumentRow(
        id=row["id"],
        filename=row["filename"],
        sha256=row["sha256"],
        num_pages=row["num_pages"],
        num_chunks=row["num_chunks"],
        ocr_pages=row["ocr_pages"],
        status=row["status"],
        is_deleted=row["is_deleted"],
        deleted_at=row["deleted_at"],
        ingested_at=row["ingested_at"],
    )


def insert_document(
    conn: sqlite3.Connection,
    *,
    filename: str,
    sha256: str,
    num_pages: int,
    num_chunks: int,
    ocr_pages: int = 0,
) -> int:
    """Inserts a new document record into the database with 'processing' status."""
    cur = conn.execute(
        "INSERT INTO documents (filename, sha256, num_pages, num_chunks, ocr_pages, status, is_deleted, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (filename, sha256, num_pages, num_chunks, ocr_pages, "processing", 0, _now()),
    )
    return int(cur.lastrowid)


def insert_chunks(conn: sqlite3.Connection, doc_id: int, chunks: list[Chunk]) -> None:
    """Inserts multiple chunk records associated with a document.

    Note: 'embedding_row' is initialized to NULL and updated only after the
    embeddings are generated and persisted.
    """
    conn.executemany(
        "INSERT INTO chunks (doc_id, ordinal, page, bbox_x0, bbox_y0, bbox_x1, bbox_y1, text, token_count, embedding_row, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                doc_id,
                index,
                chunk.page,
                chunk.bbox[0],
                chunk.bbox[1],
                chunk.bbox[2],
                chunk.bbox[3],
                chunk.text,
                chunk.token_count,
                None,
                chunk.source,
            )
            for index, chunk in enumerate(chunks)
        ],
    )


def update_chunk_embedding_rows(conn: sqlite3.Connection, doc_id: int, base_row: int) -> None:
    """Updates the 'embedding_row' for all chunks of a document.

    The 'embedding_row' serves as a direct index into the global NumPy matrix
    of embeddings. Chunks for a single document are assigned contiguous rows.
    """
    conn.execute(
        "UPDATE chunks SET embedding_row = ? + ordinal WHERE doc_id = ?",
        (base_row, doc_id),
    )


def update_document_status(conn: sqlite3.Connection, doc_id: int, status: str) -> None:
    """Updates the ingestion status of a document (e.g., 'ready', 'failed')."""
    conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))


def get_document_by_sha256(conn: sqlite3.Connection, sha256: str) -> DocumentRow | None:
    """Retrieves a document by its SHA-256 hash to avoid duplicate ingestion."""
    row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return _to_document_row(row)


def max_ready_row(conn: sqlite3.Connection) -> int:
    """Returns the highest 'embedding_row' across all successfully ingested documents.

    This is used to determine where to append new embeddings in the matrix and
    to ensure consistency during recovery.
    """
    row = conn.execute(
        "SELECT MAX(c.embedding_row) AS m FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE d.status = 'ready' AND d.is_deleted = 0"
    ).fetchone()
    return -1 if row["m"] is None else int(row["m"])


def mark_processing_and_failed_as_failed(conn: sqlite3.Connection) -> list[int]:
    """Resets all documents in 'processing' or 'failed' state back to 'failed'.

    This also clears 'embedding_row' for their chunks to prevent stale indices
    from being used in searches.
    """
    rows = conn.execute(
        "SELECT id FROM documents WHERE status IN ('processing', 'failed')"
    ).fetchall()
    ids = [int(row["id"]) for row in rows]
    for doc_id in ids:
        conn.execute("UPDATE chunks SET embedding_row = NULL WHERE doc_id = ?", (doc_id,))
        conn.execute("UPDATE documents SET status = 'failed' WHERE id = ?", (doc_id,))
    return ids


def fetch_ready_chunks_for_rebuild(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Retrieves text and embedding row mapping for all ready documents.

    Used primarily for rebuilding the BM25 keyword index from scratch.
    """
    rows = conn.execute(
        "SELECT c.embedding_row, c.text FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE d.status = 'ready' AND d.is_deleted = 0 AND c.embedding_row IS NOT NULL "
        "ORDER BY c.embedding_row"
    ).fetchall()
    return [(int(row["embedding_row"]), str(row["text"])) for row in rows]


def ready_row_set(conn: sqlite3.Connection) -> set[int]:
    """Returns the set of all valid 'embedding_row' indices for searching.

    Only chunks from 'ready' and non-deleted documents are considered.
    """
    rows = conn.execute(
        "SELECT c.embedding_row FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE d.status = 'ready' AND d.is_deleted = 0 AND c.embedding_row IS NOT NULL"
    ).fetchall()
    return {int(row["embedding_row"]) for row in rows}


def resolve_document_filter(conn: sqlite3.Connection, ids: list[int]) -> tuple[list[int], list[int]]:
    """Separates a list of document IDs into 'valid' (ready/active) and 'missing'.

    This is used during query time to handle document-level filtering.
    """
    if not ids:
        return [], []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM documents WHERE id IN ({placeholders}) AND status = 'ready' AND is_deleted = 0",
        ids,
    ).fetchall()
    valid = [int(row["id"]) for row in rows]
    missing = [doc_id for doc_id in ids if doc_id not in valid]
    return valid, missing


def row_set_for_documents(conn: sqlite3.Connection, doc_ids: list[int]) -> set[int]:
    """Returns the set of 'embedding_row' indices belonging to specific documents."""
    if not doc_ids:
        return set()
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT c.embedding_row FROM chunks c JOIN documents d ON d.id = c.doc_id "
        f"WHERE d.id IN ({placeholders}) AND d.status = 'ready' AND d.is_deleted = 0 "
        f"AND c.embedding_row IS NOT NULL",
        doc_ids,
    ).fetchall()
    return {int(row["embedding_row"]) for row in rows}
