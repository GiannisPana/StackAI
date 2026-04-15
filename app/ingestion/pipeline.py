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
import sqlite3
from typing import Any

from app.config import get_settings
from app.deps import get_store
from app.ingestion.chunker import chunk_pages
from app.ingestion.ocr_fallback import apply_ocr_fallback
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


def _cleanup_tmp_files(*paths: object) -> None:
    """Best-effort removal of staged temp files after a failed publish."""
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _mark_failed(doc_id: int, *, conn: sqlite3.Connection | None = None) -> None:
    """Set document status to failed and clean up pending chunk metadata."""
    target = conn if conn is not None else get_connection()
    try:
        with transaction(target):
            target.execute("UPDATE chunks SET embedding_row = NULL WHERE doc_id = ?", (doc_id,))
            update_document_status(target, doc_id, "failed")
    finally:
        if conn is None:
            target.close()


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
        pages, ocr_pages, ocr_page_nums = apply_ocr_fallback(client, pdf_bytes, pages)
        
        chunks = chunk_pages(
            pages,
            max_tokens=settings.max_tokens_per_chunk,
            overlap=settings.chunk_overlap_tokens,
        )
        
        if not chunks:
            return {"status": "failed", "reason": "no extractable text", "filename": filename}

        # Mark OCR-sourced chunks
        if ocr_page_nums:
            from dataclasses import replace
            chunks = [
                replace(c, source="ocr") if c.page in ocr_page_nums else c 
                for c in chunks
            ]
            
        embeddings = embed_texts(client, [chunk.text for chunk in chunks])
    except Exception as exc:
        return {"status": "failed", "reason": f"parsing/embedding: {exc}", "filename": filename}

    # PHASE 2: Staging and Atomic Publishing
    with store.writer_lock:
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

            # Step A: Database entry (Document and Chunks)
            with transaction(conn):
                doc_id = insert_document(
                    conn,
                    filename=filename,
                    sha256=sha,
                    num_pages=len(pages),
                    num_chunks=len(chunks),
                    ocr_pages=ocr_pages,
                )
                insert_chunks(conn, doc_id, chunks)

            # Step B: Stage Vector Matrix update
            base_row = store.embeddings.shape[0]
            new_matrix = build_concat_matrix(store.embeddings, embeddings)
            matrix_tmp = stage_matrix(settings.embeddings_path, new_matrix)

            # Step C: Stage BM25 Index update
            bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
            for row_id, text in fetch_ready_chunks_for_rebuild(conn):
                bm25.add(row_id, text)
            for offset, chunk in enumerate(chunks):
                bm25.add(base_row + offset, chunk.text)
            bm25.finalize()
            bm25_tmp = save_bm25(
                settings.bm25_path,
                bm25,
                publish=False,
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
        except Exception as exc:
            _cleanup_tmp_files(matrix_tmp, bm25_tmp)
            if doc_id is not None:
                _mark_failed(doc_id, conn=conn)
                store.ready_rows = ready_row_set(conn)
            return {"status": "failed", "reason": f"publish: {exc}", "filename": filename}
        finally:
            conn.close()

    return {
        "status": "ready",
        "document_id": doc_id,
        "filename": filename,
        "sha256": sha,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "ocr_pages": ocr_pages,
    }
