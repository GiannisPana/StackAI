"""Hybrid OCR fallback for low-text PDF pages.

Strategy:
- If no pages are low-text → return inputs unchanged, no OCR call.
- If ALL pages are low-text → one whole-PDF OCR call (preserves Mistral's
  cross-page layout context and minimises API calls for fully-scanned docs).
- Otherwise (mixed) → for each low-text page, extract it into its own
  single-page PDF and OCR individually. Digital pages are kept untouched.

Whole-PDF OCR is the right default for fully-scanned documents: Mistral OCR
sees the full layout context and only one API call is needed. Per-page OCR is
necessary for mixed documents to avoid clobbering high-quality digital text
with OCR output and to avoid paying for pages that don't need OCR.
"""
from __future__ import annotations

import fitz

from app.ingestion.pdf_parser import Block, PageContent, is_low_text_page
from app.mistral_client import MistralProtocol

_DEFAULT_BBOX = (0.0, 0.0, 612.0, 792.0)  # A4 fallback when no real bbox available


def _replace_with_ocr(page: PageContent, ocr_text: str) -> PageContent:
    """Return a new PageContent with blocks replaced by a single OCR text block."""
    return PageContent(
        page_num=page.page_num,
        blocks=[Block(text=ocr_text, bbox=_DEFAULT_BBOX, font_size=11.0)],
        raw_text=ocr_text,
    )


def _single_page_pdf_bytes(pdf_bytes: bytes, page_index: int) -> bytes:
    """Extract one page (0-based index) into its own PDF bytes."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = fitz.open()
        try:
            out.insert_pdf(src, from_page=page_index, to_page=page_index)
            return out.tobytes()
        finally:
            out.close()
    finally:
        src.close()


def _per_page_ocr(
    client: MistralProtocol,
    pdf_bytes: bytes,
    pages: list[PageContent],
    low_indices: list[int],
) -> tuple[list[PageContent], int, set[int]]:
    """OCR each low-text page individually using a single-page PDF payload."""
    new_pages = list(pages)
    ocr_page_nums: set[int] = set()
    for i in low_indices:
        single_pdf = _single_page_pdf_bytes(pdf_bytes, i)
        ocr_text = (client.ocr(single_pdf) or "").strip()
        if not ocr_text:
            continue
        new_pages[i] = _replace_with_ocr(pages[i], ocr_text)
        ocr_page_nums.add(pages[i].page_num)
    return new_pages, len(ocr_page_nums), ocr_page_nums


def apply_ocr_fallback(
    client: MistralProtocol,
    pdf_bytes: bytes,
    pages: list[PageContent],
) -> tuple[list[PageContent], int, set[int]]:
    """Run OCR on low-text pages using the hybrid whole-PDF / per-page strategy.

    Args:
        client: Mistral client with an ``ocr(pdf_bytes) -> str`` method.
        pdf_bytes: Raw bytes of the original PDF document.
        pages: Parsed pages from ``parse_pdf``.

    Returns:
        ``(updated_pages, ocr_page_count, ocr_page_nums)`` where
        ``ocr_page_nums`` is the set of 1-based page numbers whose blocks were
        replaced with OCR output. Non-OCR pages are returned unchanged.
    """
    low_indices = [i for i, p in enumerate(pages) if is_low_text_page(p)]
    if not low_indices:
        return pages, 0, set()

    # --- Whole-PDF path: every page is scanned ---
    if len(low_indices) == len(pages):
        ocr_text = (client.ocr(pdf_bytes) or "").strip()
        if not ocr_text:
            return pages, 0, set()

        if len(pages) == 1:
            page_texts = [ocr_text]
        else:
            split = [part.strip() for part in ocr_text.split("\f")]
            if len(split) != len(pages):
                # Provider didn't honour page boundaries — fall back to per-page.
                return _per_page_ocr(client, pdf_bytes, pages, low_indices)
            page_texts = split

        new_pages = list(pages)
        ocr_page_nums: set[int] = set()
        for i in low_indices:
            text = page_texts[i].strip()
            if not text:
                continue
            new_pages[i] = _replace_with_ocr(pages[i], text)
            ocr_page_nums.add(pages[i].page_num)
        return new_pages, len(ocr_page_nums), ocr_page_nums

    # --- Per-page path: mixed digital + scanned document ---
    return _per_page_ocr(client, pdf_bytes, pages, low_indices)
