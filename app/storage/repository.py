from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class DocumentRow:
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
    return datetime.now(timezone.utc).isoformat()


def _to_document_row(row: sqlite3.Row | None) -> DocumentRow | None:
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
    cur = conn.execute(
        "INSERT INTO documents (filename, sha256, num_pages, num_chunks, ocr_pages, status, is_deleted, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (filename, sha256, num_pages, num_chunks, ocr_pages, "processing", 0, _now()),
    )
    return int(cur.lastrowid)


def insert_chunks(conn: sqlite3.Connection, doc_id: int, chunks: list[Chunk]) -> None:
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
    conn.execute(
        "UPDATE chunks SET embedding_row = ? + ordinal WHERE doc_id = ?",
        (base_row, doc_id),
    )


def update_document_status(conn: sqlite3.Connection, doc_id: int, status: str) -> None:
    conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))


def get_document_by_sha256(conn: sqlite3.Connection, sha256: str) -> DocumentRow | None:
    row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return _to_document_row(row)


def max_ready_row(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(c.embedding_row) AS m FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE d.status = 'ready' AND d.is_deleted = 0"
    ).fetchone()
    return -1 if row["m"] is None else int(row["m"])
