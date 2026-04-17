"""
Unit tests for the PDF parsing component.

These tests verify that the PDF parser correctly extracts text blocks and
identifies page metadata from PDF documents.
"""

from __future__ import annotations

import fitz

from app.ingestion.pdf_parser import PageContent, Block, is_low_text_page, parse_pdf


def _make_pdf_bytes(pages_text: list[str]) -> bytes:
    """
    Helper function to create a minimal PDF in memory using PyMuPDF.
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            # Insert text at a standard position (72, 72)
            page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_single_page_returns_blocks():
    """
    Verify that parse_pdf correctly extracts text blocks from a single-page PDF.
    """
    pdf = _make_pdf_bytes(["hello world from page one"])
    pages = parse_pdf(pdf)

    # Should have one page
    assert len(pages) == 1
    assert pages[0].page_num == 1
    # Should contain a block with the expected text
    assert any("hello world" in block.text for block in pages[0].blocks)


def test_parse_empty_page_has_empty_blocks():
    """
    Verify that an empty PDF page is handled correctly.
    It should result in zero blocks and be identified as a low-text page.
    """
    pdf = _make_pdf_bytes([""])
    pages = parse_pdf(pdf)

    assert len(pages) == 1
    assert is_low_text_page(pages[0]) is True


def test_parse_multi_page():
    """
    Verify that multiple pages in a PDF are correctly parsed and numbered.
    """
    pdf = _make_pdf_bytes(["page one text", "page two text"])
    pages = parse_pdf(pdf)

    # Should have two pages with correct sequence
    assert [page.page_num for page in pages] == [1, 2]
