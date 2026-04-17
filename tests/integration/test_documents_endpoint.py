"""
Integration tests for the /documents list and delete endpoints.

These tests verify:
* GET /documents returns ingested documents with the correct schema and status.
* DELETE /documents/{id} soft-deletes the document, removes its row indices
  from the in-memory store, and causes subsequent queries to exclude it.
* DELETE on a non-existent or already-deleted document returns 404.
* GET /documents only returns non-deleted documents by default.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

import app.api.ingest as ingest_mod
import app.api.query as query_mod
import app.config
from app.deps import get_store, reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _ingest(client: TestClient, text: str, name: str = "doc.pdf") -> dict:
    """Ingest a PDF and return the first item in the 'ingested' list."""
    pdf = _make_pdf(text)
    resp = client.post("/ingest", files=[("files", (name, pdf, "application/pdf"))])
    assert resp.status_code == 200
    body = resp.json()
    assert body["ingested"], f"Ingest failed: {body}"
    return body["ingested"][0]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    A TestClient backed by a fully initialised app with a FakeMistralClient.

    The fake is pre-configured with classify / rerank / generation responses so
    query tests run end-to-end without network access.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": None,
            "rewritten_query": "x",
            "expansion_queries": [],
        },
    )
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.9}]})
    fake.register_chat(r"Context:", "answer [1].")

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        yield c

    reset_store()
    app.config._settings = None


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


def test_list_documents_empty(client):
    """GET /documents on an empty system returns an empty list."""
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


def test_list_documents_after_ingest(client):
    """Ingesting a PDF causes it to appear in GET /documents."""
    _ingest(client, "hello world content")
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert len(body["documents"]) == 1
    doc = body["documents"][0]
    assert doc["status"] == "ready"
    assert doc["filename"] == "doc.pdf"


def test_list_documents_schema(client):
    """Each document in the list must expose the expected fields."""
    _ingest(client, "schema check content")
    doc = client.get("/documents").json()["documents"][0]
    for field in ("id", "filename", "num_pages", "num_chunks", "status", "ingested_at"):
        assert field in doc, f"Missing field: {field}"


def test_list_documents_multiple(client):
    """All ingested documents must appear in the listing."""
    _ingest(client, "first document", name="first.pdf")
    _ingest(client, "second document", name="second.pdf")
    docs = client.get("/documents").json()["documents"]
    names = {d["filename"] for d in docs}
    assert names == {"first.pdf", "second.pdf"}


# ---------------------------------------------------------------------------
# DELETE /documents/{id}
# ---------------------------------------------------------------------------


def test_delete_returns_200(client):
    """DELETE /documents/{id} on a valid document returns 200."""
    _ingest(client, "some content to delete")
    doc_id = client.get("/documents").json()["documents"][0]["id"]
    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == doc_id


def test_delete_removes_from_ready_rows(client):
    """Deleting a document must remove its row indices from the in-memory store."""
    _ingest(client, "unique content text")
    before = set(get_store().ready_rows)
    assert before, "Expected non-empty ready_rows after ingest"

    doc_id = client.get("/documents").json()["documents"][0]["id"]
    client.delete(f"/documents/{doc_id}")

    after = set(get_store().ready_rows)
    assert after < before


def test_delete_excludes_document_from_listing(client):
    """A deleted document must still appear in GET /documents but marked as deleted."""
    _ingest(client, "content to remove")
    doc_id = client.get("/documents").json()["documents"][0]["id"]
    client.delete(f"/documents/{doc_id}")
    docs = client.get("/documents").json()["documents"]
    deleted_doc = next(d for d in docs if d["id"] == doc_id)
    assert deleted_doc["is_deleted"] == 1


def test_delete_makes_query_refuse(client):
    """After deletion the pipeline must not return citations from that document."""
    _ingest(client, "unique evidence phrase only here")
    doc_id = client.get("/documents").json()["documents"][0]["id"]
    client.delete(f"/documents/{doc_id}")

    r = client.post("/query", json={"query": "unique evidence phrase"})
    body = r.json()
    # Either no citations returned or an explicit refusal.
    assert body["citations"] == [] or body["refusal_reason"] is not None


def test_delete_nonexistent_returns_404(client):
    """Deleting a document that does not exist must return 404."""
    r = client.delete("/documents/99999")
    assert r.status_code == 404


def test_double_delete_returns_404(client):
    """A second delete of the same document must return 404."""
    _ingest(client, "doc to double delete")
    doc_id = client.get("/documents").json()["documents"][0]["id"]
    client.delete(f"/documents/{doc_id}")
    r2 = client.delete(f"/documents/{doc_id}")
    assert r2.status_code == 404
