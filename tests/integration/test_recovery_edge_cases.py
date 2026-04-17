"""
Integration tests for recovery edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest
from app.config import get_settings
from app.storage.db import init_schema, get_connection
from app.storage.vector_store import save_matrix_atomic
from app.storage.recovery import run_recovery
from app.deps import Store, set_store, reset_store
from tests.fakes.mistral import FakeMistralClient
from app.ingestion.pipeline import ingest_pdf

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    import app.config as app_config
    app_config._settings = None
    init_schema()
    s = get_settings()
    store = Store(embeddings=np.zeros((0, s.embedding_dim), dtype=np.float32))
    set_store(store)
    yield s
    reset_store()
    app_config._settings = None

def test_recovery_matrix_shorter_than_db(env):
    """Corrupt the matrix by truncating it, then verify recovery rolls back DB."""
    fake = FakeMistralClient(dim=8)
    
    # Ingest 2 documents
    pdf1 = b"%PDF-1.4\n%1.pdf\n" # We need real enough PDF for parser
    import fitz
    def _pdf(text):
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), text)
        b = doc.tobytes()
        doc.close()
        return b
    
    ingest_pdf(fake, filename="1.pdf", pdf_bytes=_pdf("text one"))
    ingest_pdf(fake, filename="2.pdf", pdf_bytes=_pdf("text two"))
    
    conn = get_connection()
    count_ready = conn.execute("SELECT COUNT(*) FROM documents WHERE status='ready'").fetchone()[0]
    assert count_ready == 2
    
    # Manually truncate the matrix to only have rows for document 1
    # Find how many chunks doc 1 has
    doc1_chunks = conn.execute("SELECT num_chunks FROM documents WHERE filename='1.pdf'").fetchone()[0]
    
    matrix = np.load(env.embeddings_path)
    truncated = matrix[:doc1_chunks]
    save_matrix_atomic(env.embeddings_path, truncated)
    
    # Run recovery
    run_recovery()
    
    # Doc 2 should now be failed
    doc2_status = conn.execute("SELECT status FROM documents WHERE filename='2.pdf'").fetchone()[0]
    assert doc2_status == "failed"
    
    # Doc 1 should still be ready
    doc1_status = conn.execute("SELECT status FROM documents WHERE filename='1.pdf'").fetchone()[0]
    assert doc1_status == "ready"
    
    conn.close()
