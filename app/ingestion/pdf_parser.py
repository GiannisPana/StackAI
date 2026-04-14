"""
PDF parsing utilities.

This module provides functions for extracting text blocks and page metadata
from PDF documents using PyMuPDF (fitz).
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz


@dataclass(frozen=True)
class Block:
    """
    A block of text on a PDF page with its layout metadata.

    Attributes:
        text: The text content of the block.
        bbox: (x0, y0, x1, y1) bounding box of the block.
        font_size: Average font size of the text within the block.
    """
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float


@dataclass(frozen=True)
class PageContent:
    """
    Representation of a single PDF page's content.

    Attributes:
        page_num: 1-based page number.
        blocks: List of text blocks extracted from the page.
        raw_text: Full raw text of the page.
    """
    page_num: int
    blocks: list[Block]
    raw_text: str


def parse_pdf(pdf_bytes: bytes) -> list[PageContent]:
    """
    Parse a PDF from bytes into a list of PageContent objects.

    Extracts text blocks and their font sizes to enable layout-aware chunking.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[PageContent] = []
        for index, page in enumerate(doc):
            blocks: list[Block] = []
            raw_text = page.get_text("text") or ""
            text_dict = page.get_text("dict")
            for raw_block in text_dict.get("blocks", []):
                # Type 0 is text block in PyMuPDF
                if raw_block.get("type", 0) != 0:
                    continue
                spans: list[str] = []
                sizes: list[float] = []
                for line in raw_block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if text:
                            spans.append(text)
                            sizes.append(float(span.get("size", 0.0)))
                block_text = " ".join(spans).strip()
                if not block_text:
                    continue
                bbox = tuple(float(value) for value in raw_block.get("bbox", (0, 0, 0, 0)))
                avg_font_size = sum(sizes) / len(sizes) if sizes else 0.0
                blocks.append(Block(text=block_text, bbox=bbox, font_size=avg_font_size))
            pages.append(PageContent(page_num=index + 1, blocks=blocks, raw_text=raw_text))
        return pages
    finally:
        doc.close()


def is_low_text_page(page: PageContent, min_chars: int = 50) -> bool:
    """True if the page has very little extractable text (likely scanned)."""
    return len(page.raw_text.strip()) < min_chars
