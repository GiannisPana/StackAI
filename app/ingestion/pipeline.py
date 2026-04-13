from __future__ import annotations

import hashlib
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
    get_document_by_sha256,
    insert_chunks,
    insert_document,
    update_chunk_embedding_rows,
    update_document_status,
)
from app.storage.vector_store import concat_and_save


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_pdf(client: MistralProtocol, *, filename: str, pdf_bytes: bytes) -> dict[str, Any]:
    settings = get_settings()
    store = get_store()
    sha = _sha256(pdf_bytes)

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

    pages = parse_pdf(pdf_bytes)
    chunks = chunk_pages(
        pages,
        max_tokens=settings.max_tokens_per_chunk,
        overlap=settings.chunk_overlap_tokens,
    )
    embeddings = embed_texts(client, [chunk.text for chunk in chunks])

    conn = get_connection()
    try:
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
            base_row = store.embeddings.shape[0]
            new_matrix = concat_and_save(settings.embeddings_path, store.embeddings, embeddings)

            bm25 = store.bm25 if isinstance(store.bm25, BM25Index) else BM25Index()
            for offset, chunk in enumerate(chunks):
                bm25.add(base_row + offset, chunk.text)
            bm25.finalize()
            save_bm25(settings.bm25_path, bm25)

            with transaction(conn):
                update_chunk_embedding_rows(conn, doc_id, base_row=base_row)
                update_document_status(conn, doc_id, "ready")

            store.embeddings = new_matrix
            store.bm25 = bm25
            store.ready_rows |= set(range(base_row, base_row + len(chunks)))

        return {
            "status": "ready",
            "document_id": doc_id,
            "filename": filename,
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "ocr_pages": 0,
        }
    except Exception:
        with transaction(conn):
            existing = get_document_by_sha256(conn, sha)
            if existing is not None:
                update_document_status(conn, existing.id, "failed")
        raise
    finally:
        conn.close()
