"""
Unit tests for the hybrid OCR fallback helper.

Covers the three paths: no-op, whole-PDF, per-page, and the split-disagreement
fallback within the whole-PDF path.
"""
from __future__ import annotations

import fitz
import pytest

from app.ingestion.ocr_fallback import _single_page_pdf_bytes, apply_ocr_fallback
from app.ingestion.pdf_parser import Block, PageContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(page_num: int, text: str = "") -> PageContent:
    bbox = (0.0, 0.0, 612.0, 792.0)
    blocks = [Block(text=text, bbox=bbox, font_size=11.0)] if text else []
    return PageContent(page_num=page_num, blocks=blocks, raw_text=text)


def _scanned_page(page_num: int) -> PageContent:
    """A blank page that triggers is_low_text_page (< 50 chars)."""
    return _make_page(page_num, "")


def _digital_page(page_num: int) -> PageContent:
    return _make_page(page_num, "A" * 100)


class _FakeClient:
    """Minimal fake that records calls and returns responses by call index.

    Because fitz embeds a random document ID in PDF bytes, the bytes produced
    by _single_page_pdf_bytes in the test setup differ from those produced
    inside apply_ocr_fallback.  We therefore match responses by call order
    (index) rather than byte equality.
    """

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0
        self.ocr_calls: list[bytes] = []

    def ocr(self, pdf_bytes: bytes) -> str:
        self.ocr_calls.append(pdf_bytes)
        response = self._responses[self._idx] if self._idx < len(self._responses) else ""
        self._idx += 1
        return response

    # satisfy MistralProtocol — not needed for these tests
    def embed(self, text: str):  # type: ignore[override]
        raise NotImplementedError

    def embed_batch(self, texts):  # type: ignore[override]
        raise NotImplementedError

    def chat(self, messages, response_format=None, temperature=None):  # type: ignore[override]
        raise NotImplementedError


def _make_single_page_scanned_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _make_two_page_scanned_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_low_text_returns_unchanged_and_no_ocr_call():
    pages = [_digital_page(1), _digital_page(2)]
    fake = _FakeClient([])
    result, count, nums = apply_ocr_fallback(fake, b"any", pages)
    assert result is pages
    assert count == 0
    assert nums == set()
    assert fake.ocr_calls == [], "ocr() must not be called when all pages have text"


def test_all_pages_scanned_uses_single_whole_pdf_call():
    pdf = _make_two_page_scanned_pdf()
    pages = [_scanned_page(1), _scanned_page(2)]
    fake = _FakeClient(["Page one\fPage two"])

    result, count, nums = apply_ocr_fallback(fake, pdf, pages)

    assert len(fake.ocr_calls) == 1, "Whole-PDF path must use exactly one OCR call"
    assert fake.ocr_calls[0] == pdf, "Whole-PDF path must pass original pdf_bytes"
    assert count == 2
    assert nums == {1, 2}
    assert result[0].raw_text == "Page one"
    assert result[1].raw_text == "Page two"


def test_single_page_scanned_whole_pdf_path():
    pdf = _make_single_page_scanned_pdf()
    pages = [_scanned_page(1)]
    fake = _FakeClient(["Only page OCR text"])

    result, count, nums = apply_ocr_fallback(fake, pdf, pages)

    assert count == 1
    assert nums == {1}
    assert result[0].raw_text == "Only page OCR text"
    assert len(fake.ocr_calls) == 1


def test_mixed_pdf_uses_per_page_calls():
    """Only the blank page (index 1) should be OCR'd; digital page stays untouched."""
    pdf = _make_two_page_scanned_pdf()
    pages = [_digital_page(1), _scanned_page(2)]
    # Single response for the one per-page call
    fake = _FakeClient(["OCR for scanned page"])

    result, count, nums = apply_ocr_fallback(fake, pdf, pages)

    assert len(fake.ocr_calls) == 1, "Only one per-page OCR call (for page 2)"
    # The call must be for a single-page PDF (smaller than the original 2-page PDF)
    assert len(fake.ocr_calls[0]) < len(pdf), "Per-page PDF should be smaller than source"
    assert count == 1
    assert nums == {2}
    # Page 1 is untouched
    assert result[0].raw_text == "A" * 100
    # Page 2 replaced with OCR
    assert result[1].raw_text == "OCR for scanned page"


def test_per_page_fallback_when_split_count_disagrees():
    """If whole-PDF OCR returns wrong number of \\f sections, fall back to per-page."""
    pdf = _make_two_page_scanned_pdf()
    pages = [_scanned_page(1), _scanned_page(2)]
    # Call 0: whole-PDF with no \\f → split gives 1 section for 2 pages → fallback
    # Calls 1 & 2: per-page responses
    fake = _FakeClient(["no form feed here", "Page one per-page", "Page two per-page"])

    result, count, nums = apply_ocr_fallback(fake, pdf, pages)

    # First call is the whole-PDF attempt; then two per-page calls
    assert len(fake.ocr_calls) == 3
    assert fake.ocr_calls[0] == pdf, "First call must be the whole-PDF attempt"
    # Per-page calls are single-page PDFs (smaller than the 2-page source)
    assert len(fake.ocr_calls[1]) < len(pdf)
    assert len(fake.ocr_calls[2]) < len(pdf)
    assert result[0].raw_text == "Page one per-page"
    assert result[1].raw_text == "Page two per-page"
    assert count == 2


def test_empty_ocr_response_skips_page():
    """If OCR returns empty string for a page, that page should remain unchanged."""
    pdf = _make_single_page_scanned_pdf()
    pages = [_scanned_page(1)]
    fake = _FakeClient([""])  # empty response

    result, count, nums = apply_ocr_fallback(fake, pdf, pages)

    assert count == 0
    assert nums == set()
    assert result[0].raw_text == ""  # unchanged
