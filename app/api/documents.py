"""
Document management endpoints for the StackAI RAG application.

Provides two endpoints:

* ``GET /documents`` - list all documents with their metadata, including
  soft-deleted ones.
* ``DELETE /documents/{doc_id}`` - soft-delete a document and evict its
  embedding rows from the in-memory store so they are excluded from all
  future queries without restarting the server.

The delete operation is atomic: the SQLite update and the in-memory eviction
are performed under the store's writer lock so no search can see a half-deleted
state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import get_store
from app.storage.db import get_connection, transaction
from app.storage.repository import (
    get_document_by_id,
    list_documents,
    soft_delete_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    """Serialisable representation of a single document record."""

    id: int
    filename: str
    num_pages: int
    num_chunks: int
    ocr_pages: int
    status: str
    is_deleted: int
    ingested_at: str


class DocumentsResponse(BaseModel):
    """Response wrapper for the document listing endpoint."""

    documents: list[DocumentOut]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=DocumentsResponse)
def list_docs() -> DocumentsResponse:
    """
    List all documents in the archive.

    Returns documents ordered newest-first by ingestion time.

    Returns:
        A :class:`DocumentsResponse` containing the document list.
    """
    conn = get_connection()
    try:
        rows = list_documents(conn)
    finally:
        conn.close()

    return DocumentsResponse(
        documents=[
            DocumentOut(
                id=row.id,
                filename=row.filename,
                num_pages=row.num_pages,
                num_chunks=row.num_chunks,
                ocr_pages=row.ocr_pages,
                status=row.status,
                is_deleted=row.is_deleted,
                ingested_at=row.ingested_at,
            )
            for row in rows
        ]
    )


@router.delete("/{doc_id}")
def delete_doc(doc_id: int) -> dict:
    """
    Soft-delete a document and evict it from the search index.

    The document record is retained in the database for audit purposes but
    marked with ``is_deleted = 1``.  Its embedding row indices are
    simultaneously removed from the in-memory :class:`~app.deps.Store` so
    that subsequent queries immediately reflect the deletion.

    Args:
        doc_id: The integer primary key of the document to delete.

    Returns:
        A JSON object with ``deleted`` (the document ID) and ``removed_rows``
        (the number of embedding rows evicted from the store).

    Raises:
        HTTPException 404: If no document with this ID exists or it has
            already been deleted.
    """
    store = get_store()

    # Acquire the writer lock for the duration of the DB update + in-memory
    # eviction so no search thread can observe a partially deleted state.
    with store.writer_lock:
        conn = get_connection()
        try:
            existing = get_document_by_id(conn, doc_id)
            if existing is None or existing.is_deleted:
                raise HTTPException(status_code=404, detail="document not found")

            with transaction(conn):
                affected_rows = soft_delete_document(conn, doc_id)
        finally:
            conn.close()

        # Evict the deleted document's rows from the in-memory store.
        store.ready_rows -= set(affected_rows)

    return {"deleted": doc_id, "removed_rows": len(affected_rows)}
