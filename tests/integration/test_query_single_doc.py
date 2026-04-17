"""
Integration tests for query threshold logic on single-document corpora.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.deps import reset_store
from tests.fakes.mistral import FakeMistralClient
import app.config
import app.api.query as query_mod
import app.api.ingest as ingest_mod

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    
    import app.config as app_config
    app_config._settings = None
    fake = FakeMistralClient(dim=8)
    
    # Mock ingest to use our fake
    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    # Mock query to use our fake
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)
    
    app = create_app()
    with TestClient(app) as c:
        yield c, fake
    reset_store()
    app_config._settings = None

def test_query_single_doc_clears_threshold(client):
    """Verify that a single relevant candidate clears the threshold gate."""
    c, fake = client
    
    # 1. Ingest one doc
    import fitz
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 72), "the secret code is 12345")
    pdf = doc.tobytes()
    doc.close()
    
    c.post("/ingest", files=[("files", ("secret.pdf", pdf, "application/pdf"))])
    
    # 2. Setup fake to return a moderate score for the query
    fake.register_chat("what is the secret code", "The secret code is 12345 [1].")
    
    response = c.post("/query", json={"query": "what is the secret code?"})
    assert response.status_code == 200
    body = response.json()
    assert body.get("refusal_reason") is None
    assert len(body["citations"]) > 0
    assert "12345" in body["answer"]
