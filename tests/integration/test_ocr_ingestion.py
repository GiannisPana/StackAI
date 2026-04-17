"""
Integration tests for OCR ingestion path.
"""
from __future__ import annotations

import pytest
import fitz
from fastapi.testclient import TestClient

from app.main import create_app
from app.deps import reset_store
from tests.fakes.mistral import FakeMistralClient
import app.config
import app.api.ingest as ingest_mod

def _scanned_pdf(*, pages: int = 1) -> bytes:
    """Blank PDF pages that trigger low-text OCR logic."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    
    import app.config as app_config
    app_config._settings = None
    fake = FakeMistralClient(dim=8)

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    app = create_app()
    with TestClient(app) as c:
        yield c, fake
    reset_store()
    app_config._settings = None

def test_ingest_scanned_pdf_triggers_ocr(client):
    c, fake = client
    pdf = _scanned_pdf()
    fake.register_ocr(pdf, "OCR recognized text")
    response = c.post("/ingest", files=[("files", ("scanned.pdf", pdf, "application/pdf"))])
    
    assert response.status_code == 200
    body = response.json()
    assert len(body["ingested"]) == 1
    assert body["ingested"][0]["ocr_pages"] == 1
    assert body["ingested"][0]["num_chunks"] == 1
    
    # Verify DB source='ocr'
    from app.storage.db import get_connection
    conn = get_connection()
    row = conn.execute("SELECT source, text FROM chunks WHERE doc_id=?", (body["ingested"][0]["document_id"],)).fetchone()
    conn.close()
    assert row["source"] == "ocr"
    assert "OCR recognized text" in row["text"]


def _mixed_pdf() -> bytes:
    """Page 1 has born-digital text; page 2 is blank (scanned)."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Digital page one with real born-digital text content.", fontsize=11)
    doc.new_page()  # blank page → triggers OCR
    data = doc.tobytes()
    doc.close()
    return data


def _single_page_pdf_bytes(full_pdf: bytes, page_index: int) -> bytes:
    """Extract one page into its own PDF (mirrors what apply_ocr_fallback does for per-page OCR)."""
    src = fitz.open(stream=full_pdf, filetype="pdf")
    out = fitz.open()
    out.insert_pdf(src, from_page=page_index, to_page=page_index)
    data = out.tobytes()
    out.close()
    src.close()
    return data


def test_mixed_pdf_only_scanned_page_is_ocred(client):
    """Digital pages must keep their original text; only blank pages go through OCR."""
    c, fake = client
    pdf = _mixed_pdf()
    # Per-page OCR generates a new 1-page PDF internally (different bytes each time).
    # Use register_ocr_by_page_count so the fake matches on page count, not exact bytes.
    fake.register_ocr_by_page_count(1, "OCR text for page two only")

    response = c.post("/ingest", files=[("files", ("mixed.pdf", pdf, "application/pdf"))])
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"][0]["ocr_pages"] == 1  # only page 2

    from app.storage.db import get_connection
    conn = get_connection()
    rows = conn.execute(
        "SELECT page, source, text FROM chunks WHERE doc_id = ? ORDER BY page, ordinal",
        (body["ingested"][0]["document_id"],),
    ).fetchall()
    conn.close()

    by_page: dict[int, tuple[str, str]] = {}
    for row in rows:
        by_page.setdefault(int(row["page"]), (row["source"], ""))
        src_val, txt = by_page[int(row["page"])]
        by_page[int(row["page"])] = (src_val, txt + str(row["text"]))

    assert by_page[1][0] == "pdf_text", "Page 1 must keep digital source"
    assert "born-digital" in by_page[1][1], "Page 1 must retain original text"
    assert by_page[2][0] == "ocr", "Page 2 must be marked as ocr"
    assert "OCR text for page two only" in by_page[2][1]


def test_multi_page_ocr_maps_page_text_without_duplication(client):
    """OCR text for a multi-page scan must not be copied verbatim onto every page."""
    c, fake = client
    pdf = _scanned_pdf(pages=2)
    fake.register_ocr(pdf, "First page OCR\fSecond page OCR")

    response = c.post("/ingest", files=[("files", ("scanned.pdf", pdf, "application/pdf"))])

    assert response.status_code == 200
    body = response.json()
    assert body["ingested"][0]["ocr_pages"] == 2

    from app.storage.db import get_connection

    conn = get_connection()
    rows = conn.execute(
        "SELECT page, source, text FROM chunks WHERE doc_id = ? ORDER BY page, ordinal",
        (body["ingested"][0]["document_id"],),
    ).fetchall()
    conn.close()

    texts_by_page: dict[int, str] = {}
    for row in rows:
        assert row["source"] == "ocr"
        texts_by_page.setdefault(int(row["page"]), "")
        texts_by_page[int(row["page"])] += str(row["text"])

    assert "First page OCR" in texts_by_page[1]
    assert "Second page OCR" not in texts_by_page[1]
    assert "Second page OCR" in texts_by_page[2]
    assert "First page OCR" not in texts_by_page[2]
