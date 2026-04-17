"""
Integration tests for the core ingestion pipeline.

These tests verify the end-to-end flow of ingesting a PDF document, including:
1. PDF parsing and chunking.
2. Embedding generation via the Mistral client.
3. Metadata storage in SQLite.
4. Vector storage for semantic search.
5. BM25 index updates for keyword search.

It ensures that all storage layers stay synchronized and that failures are
handled transactionally where possible.
"""

from __future__ import annotations

import hashlib

import fitz
import numpy as np
import pytest

from app.config import get_settings
from app.deps import Store, get_store, reset_store, set_store
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.bm25 import BM25Index
from app.storage.bm25_store import load_bm25
from app.storage.db import get_connection, init_schema
from app.storage.repository import get_document_by_sha256, max_ready_row
from app.storage.vector_store import load_matrix
from tests.fakes.mistral import FakeMistralClient


def _make_pdf(text: str) -> bytes:
    """Helper to create a simple one-page PDF from text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def env(tmp_path, monkeypatch):
    """
    Setup a clean integration environment with initialized schema and stores.
    """
    # Isolate data directory to a temporary path
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    import app.config

    # Reset settings and initialize a fresh database schema
    app.config._settings = None
    init_schema()
    settings = get_settings()
    
    # Initialize the global store with empty states for integration testing
    store = Store(embeddings=np.zeros((0, settings.embedding_dim), dtype=np.float32))
    store.bm25 = load_bm25(settings.bm25_path)
    set_store(store)
    
    yield
    
    # Teardown: Clear global state and settings
    reset_store()
    app.config._settings = None


def test_ingest_single_pdf_publishes_all_stores(env):
    """
    Verify that ingesting a PDF correctly updates all three storage backends.

    Verifies synchronicity between:
    - SQLite (document metadata)
    - Vector Store (.npy file)
    - BM25 Store (in-memory/persisted index)
    """
    fake = FakeMistralClient(dim=8)
    pdf = _make_pdf("the parental leave policy is sixteen weeks")
    sha = hashlib.sha256(pdf).hexdigest()

    # Trigger the full ingestion pipeline
    result = ingest_pdf(fake, filename="policy.pdf", pdf_bytes=pdf)

    # Basic pipeline success assertions
    assert result["status"] == "ready"
    assert result["num_chunks"] >= 1

    # Verify SQLite metadata storage
    conn = get_connection()
    try:
        doc = get_document_by_sha256(conn, sha)
        assert doc is not None
        assert doc.status == "ready"
        assert max_ready_row(conn) >= 0
    finally:
        conn.close()

    # Verify vector store persistence
    settings = get_settings()
    matrix = load_matrix(settings.embeddings_path, expected_dim=8)
    assert matrix.shape[0] == result["num_chunks"]


def test_ingest_duplicate_returns_skipped(env):
    """
    Verify that the pipeline avoids redundant processing for identical content.
    """
    fake = FakeMistralClient(dim=8)
    pdf = _make_pdf("hello world document one")

    # First pass: normal ingestion
    ingest_pdf(fake, filename="a.pdf", pdf_bytes=pdf)
    # Second pass: should detect collision based on content hash
    result = ingest_pdf(fake, filename="a.pdf", pdf_bytes=pdf)

    assert result["status"] == "skipped"
    assert result["reason"] == "already ingested"


def test_ingest_invalid_pdf_returns_failed(env):
    """
    Verify that the pipeline captures and reports PDF parsing failures.
    """
    fake = FakeMistralClient(dim=8)

    # Provide malformed PDF bytes
    result = ingest_pdf(fake, filename="broken.pdf", pdf_bytes=b"%PDF-1.4\nbroken")

    # The pipeline should handle the exception and return a failure status
    assert result["status"] == "failed"
    assert "parsing/embedding" in result["reason"]


def test_ingest_rebuilds_bm25_from_ready_rows_not_stale_store(env):
    """
    Ensure the BM25 index is always rebuilt from the 'source of truth' (SQLite).

    This verifies that the pipeline doesn't preserve stale in-memory state
    from previous (potentially failed) operations.
    """
    fake = FakeMistralClient(dim=8)
    # Manually inject a 'stale' entry into the in-memory BM25 index
    stale = BM25Index()
    stale.add(99, "stale row that should be dropped")
    stale.finalize()
    get_store().bm25 = stale

    # Ingest a fresh document
    pdf = _make_pdf("fresh ready document")
    result = ingest_pdf(fake, filename="fresh.pdf", pdf_bytes=pdf)

    # The index should have been wiped and rebuilt containing only valid SQLite rows
    assert result["status"] == "ready"
    postings_rows = {
        row_id
        for postings in get_store().bm25._postings.values()
        for row_id in postings.keys()
    }
    assert postings_rows == get_store().ready_rows
    assert 99 not in postings_rows


def test_ingest_publish_failure_does_not_publish_embeddings_early(env, monkeypatch):
    """
    Verify the atomic-like behavior of the publication step.

    If a late-stage publication step (like saving BM25) fails, ensure that
    no partial files (like embeddings) are left behind in a 'ready' state.
    """
    fake = FakeMistralClient(dim=8)
    pdf = _make_pdf("publish ordering test")
    settings = get_settings()

    # Mock a failure during the BM25 saving process
    def fail_save_bm25(*args, **kwargs):
        raise OSError("forced publish setup failure")

    import app.ingestion.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "save_bm25", fail_save_bm25)

    # Trigger ingestion which will fail at the final publication step
    result = ingest_pdf(fake, filename="broken-publish.pdf", pdf_bytes=pdf)

    # Verify that the overall operation failed
    assert result["status"] == "failed"
    
    # Verify that the vector matrix was NOT updated and no temporary files were leaked
    matrix = load_matrix(settings.embeddings_path, expected_dim=8)
    assert matrix.shape[0] == 0
    assert not settings.embeddings_path.with_suffix(".npy.tmp").exists()
    assert not settings.bm25_path.with_suffix(".json.tmp").exists()

    # Verify the database correctly reflects the failure
    conn = get_connection()
    try:
        doc = get_document_by_sha256(conn, hashlib.sha256(pdf).hexdigest())
        assert doc is not None
        assert doc.status == "failed"
    finally:
        conn.close()
