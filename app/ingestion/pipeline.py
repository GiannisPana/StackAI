"""
Document ingestion pipeline.

This module orchestrates the multi-phase ingestion process:
1. Parsing and Chunking: Extracts content from PDFs.
2. Embedding: Generates vector representations of chunks.
3. Staging: Updates database and prepares new index files in temporary storage.
4. Publishing: Atomically swaps indexes and updates the in-memory state.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from app.config import get_settings
from app.deps import get_store
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_parser import parse_pdf
from app.mistral_client import MistralProtocol
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import embed_texts
from app.storage.bm25_store import save_bm25
from app.storage.db import get_connection, transaction
from app.storage.repository import (
    fetch_ready_chunks_for_rebuild,
    get_document_by_sha256,
    insert_chunks,
    insert_document,
    ready_row_set,
    update_chunk_embedding_rows,
    update_document_status,
)
from app.storage.vector_store import build_concat_matrix, stage_matrix


def _sha256(data: bytes) -> str:
    """Return the SHA256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _mark_failed(conn, doc_id: int) -> None:
    """Set document status to failed and clean up pending chunk metadata."""
    with transaction(conn):
        conn.execute("UPDATE chunks SET embedding_row = NULL WHERE doc_id = ?", (doc_id,))
        update_document_status(conn, doc_id, "failed")


def ingest_pdf(client: MistralProtocol, *, filename: str, pdf_bytes: bytes) -> dict[str, Any]:
    """
    Main entry point for document ingestion.

    Coordinates a fail-safe, atomic ingestion process that updates both the
    database and search indexes (Vector and BM25).
    """
    settings = get_settings()
    store = get_store()
    sha = _sha256(pdf_bytes)
    doc_id: int | None = None
    matrix_tmp = None
    bm25_tmp = None

    # PHASE 0: Deduplication check
    conn = get_connection()
    try:
        existing = get_document_by_sha256(conn, sha)
        if existing is not None and existing.status == "ready" and not existing.is_deleted:
            return {
                "status": "skipped",
                "reason": "already ingested",
                "document_id": existing.id,
                "filename": filename,
            }
    finally:
        conn.close()

    # PHASE 1: Parsing, Chunking and Embedding
    try:
        pages = parse_pdf(pdf_bytes)
    except Exception as exc:
        return {"status": "failed", "reason": f"invalid PDF: {exc}", "filename": filename}

    chunks = chunk_pages(
        pages,
        max_tokens=settings.max_tokens_per_chunk,
        overlap=settings.chunk_overlap_tokens,
    )
    embeddings = embed_texts(client, [chunk.text for chunk in chunks])

    # PHASE 2: Staging and Atomic Publishing
    conn = get_connection()
    try:
        # Step A: Preliminary database entry (Document and Chunks)
        with transaction(conn):
            doc_id = insert_document(
                conn,
                filename=filename,
                sha256=sha,
                num_pages=len(pages),
                num_chunks=len(chunks),
            )
            insert_chunks(conn, doc_id, chunks)

        with store.writer_lock:
            # Step B: Stage Vector Matrix update
            base_row = store.embeddings.shape[0]
            new_matrix = build_concat_matrix(store.embeddings, embeddings)
            matrix_tmp = stage_matrix(settings.embeddings_path, new_matrix)

            # Step C: Stage BM25 Index update
            bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
            # Rebuild index from all ready chunks to ensure consistency
            for row_id, text in fetch_ready_chunks_for_rebuild(conn):
                bm25.add(row_id, text)
            # Add newly ingested chunks
            for offset, chunk in enumerate(chunks):
                bm25.add(base_row + offset, chunk.text)
            bm25.finalize()
            bm25_tmp = save_bm25(
                settings.bm25_path,
                bm25,
                publish=False, # Write to .tmp first
            )

            # Step D: Finalize Database State
            with transaction(conn):
                update_chunk_embedding_rows(conn, doc_id, base_row=base_row)
                update_document_status(conn, doc_id, "ready")

            # Step E: Publish staged files (Atomic swap)
            os.replace(matrix_tmp, settings.embeddings_path)
            os.replace(bm25_tmp, settings.bm25_path)
            matrix_tmp = None
            bm25_tmp = None

            # Step F: Update In-Memory Cache
            store.embeddings = new_matrix
            store.bm25 = bm25
            store.ready_rows = ready_row_set(conn)

        return {
            "status": "ready",
            "document_id": doc_id,
            "filename": filename,
            "sha256": sha,
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "ocr_pages": 0,
        }
    except Exception as exc:
        # Rollback: Remove temporary files if something went wrong before publishing
        for tmp in (
            settings.embeddings_path.with_suffix(settings.embeddings_path.suffix + ".tmp"),
            settings.bm25_path.with_suffix(settings.bm25_path.suffix + ".tmp"),
        ):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        if doc_id is not None:
            _mark_failed(conn, doc_id)
        return {"status": "failed", "reason": f"publish: {exc}", "filename": filename}
    finally:
        conn.close()
