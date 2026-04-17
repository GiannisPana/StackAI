"""
Integration tests for the PDF ingestion FastAPI endpoint.

These tests verify the interaction between the /ingest endpoint, the PDF
parsing layer, and the backend ingestion pipeline. It exercises the full
request/response cycle, including file upload handling and error reporting.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from app.deps import reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient

import app.api.ingest as ingest_mod


def _pdf(text: str) -> bytes:
    """Generate a valid single-page PDF in memory for testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Setup a FastAPI TestClient with a clean environment for integration tests.

    Mocks external dependencies like the Mistral API and resets the global store
    after each test.
    """
    # Environment isolation for the test run
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    import app.config as app_config

    # Ensure clean configuration and global store state
    app_config._settings = None
    fake = FakeMistralClient(dim=8)
    # Inject fake client to avoid real network calls during integration tests
    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Teardown to maintain test isolation
    reset_store()
    app_config._settings = None


def test_ingest_single_pdf(client):
    """
    Verify the happy path for ingesting a single PDF.

    Tests the end-to-end flow from multipart/form-data upload to successful
    ingestion and response.
    """
    pdf = _pdf("the parental leave policy is sixteen weeks long")

    # Simulate multipart file upload to the ingestion endpoint
    response = client.post("/ingest", files=[("files", ("a.pdf", pdf, "application/pdf"))])

    # Assert that the pipeline correctly processed the file and returned the 'ready' status
    assert response.status_code == 200
    body = response.json()
    assert len(body["ingested"]) == 1
    assert body["ingested"][0]["status"] == "ready"


def test_ingest_duplicate_lands_in_skipped(client):
    """
    Verify that already-ingested files are correctly identified as duplicates.

    Tests the storage layer's duplicate detection logic when accessed via the endpoint.
    """
    pdf = _pdf("alpha beta gamma delta")
    # First ingestion attempt
    client.post("/ingest", files=[("files", ("a.pdf", pdf, "application/pdf"))])

    # Second ingestion attempt with the same content
    response = client.post("/ingest", files=[("files", ("a.pdf", pdf, "application/pdf"))])

    # The pipeline should detect the SHA256 collision and skip processing
    body = response.json()
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["reason"] == "already ingested"


def test_ingest_rejects_non_pdf_to_failed(client):
    """
    Verify that non-PDF files are gracefully rejected by the endpoint.

    Tests the integration between file type validation and the response schema.
    """
    # Attempt to upload a plain text file instead of a PDF
    response = client.post("/ingest", files=[("files", ("a.txt", b"not a pdf", "text/plain"))])

    # Errors should be captured in the 'failed' list rather than crashing the request
    body = response.json()
    assert len(body["failed"]) == 1
    assert "pdf" in body["failed"][0]["reason"].lower()


def test_ingest_oversized_file_goes_to_failed_not_413(client, monkeypatch):
    """
    Verify that oversized files are handled within the application logic.

    Ensures the application provides a descriptive error instead of a generic
    server-level 413 error.
    """
    # Patch settings to a very small limit so the endpoint reads one source of truth.
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: SimpleNamespace(max_pdf_bytes=100))
    big = b"%PDF-1.4\n" + b"x" * 500

    response = client.post("/ingest", files=[("files", ("big.pdf", big, "application/pdf"))])

    # Validate that the size limit check was triggered and reported
    assert response.status_code == 200
    body = response.json()
    assert len(body["failed"]) == 1
    assert "limit" in body["failed"][0]["reason"].lower()


def test_ingest_all_invalid_returns_400(client):
    """
    Verify that a 400 Bad Request is returned when no files can be processed.

    Tests the collective result logic when every item in a batch fails validation.
    """
    # Batch upload with only invalid files
    response = client.post(
        "/ingest",
        files=[
            ("files", ("a.txt", b"not pdf", "text/plain")),
            ("files", ("b.pdf", b"not a real pdf", "application/pdf")),
        ],
    )

    # All files failed, so the overall request should reflect a failure
    assert response.status_code == 400
    body = response.json()
    assert body["ingested"] == []
    assert body["skipped"] == []
    assert len(body["failed"]) == 2
