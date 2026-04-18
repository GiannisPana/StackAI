"""
Unit tests for the document chunking logic.

These tests ensure that the PDF parser's output is correctly split into manageable
chunks for retrieval while preserving context through boundaries and overlap.
"""

from __future__ import annotations

import app.ingestion.chunker as chunker
from app.ingestion.chunker import _approx_tokens, chunk_pages
from app.ingestion.pdf_parser import Block, PageContent


def _page(page_num: int, blocks: list[tuple[str, float]]) -> PageContent:
    """
    Helper function to create a PageContent object with synthetic blocks.
    """
    page_blocks = [
        Block(text=text, bbox=(0, index * 10, 100, (index + 1) * 10), font_size=size)
        for index, (text, size) in enumerate(blocks)
    ]
    return PageContent(
        page_num=page_num,
        blocks=page_blocks,
        raw_text=" ".join(text for text, _ in blocks),
    )


def test_never_spans_page_boundary():
    """
    Verify that chunks do not span across page boundaries.
    Each chunk should be contained within a single page to maintain context locality.
    """
    page1 = _page(1, [("alpha beta gamma", 11)])
    page2 = _page(2, [("delta epsilon zeta", 11)])

    # Splitting with a large token limit should still separate pages
    chunks = chunk_pages([page1, page2], max_tokens=100, overlap=10)

    # Chunks should represent both pages
    assert {chunk.page for chunk in chunks} == {1, 2}


def test_heading_splits_chunk_boundary():
    """
    Verify that blocks with significantly larger font sizes (headings) trigger a new chunk.
    This helps keep section headers grouped with their subsequent content.
    """
    page = _page(1, [("body a", 11), ("SECTION TWO", 22), ("body b", 11)])

    # "SECTION TWO" should start a new chunk
    chunks = chunk_pages([page], max_tokens=1000, overlap=0)
    texts = [chunk.text for chunk in chunks]

    # Content before the heading should be separate from content after
    assert any("body a" in text and "body b" not in text for text in texts)


def test_size_cap_splits_long_block():
    """
    Verify that very long blocks exceeding the token limit are split into multiple chunks.
    """
    long_text = " ".join(["word"] * 600)
    page = _page(1, [(long_text, 11)])

    # Splitting a 600-word block with a 100-token limit
    chunks = chunk_pages([page], max_tokens=100, overlap=20)

    # Should result in multiple chunks
    assert len(chunks) >= 6
    # Each chunk should respect the token limit (with a small buffer for overlap)
    for chunk in chunks:
        assert _approx_tokens(chunk.text) <= 130


def test_overlap_present_between_consecutive_chunks():
    """
    Verify that there is text overlap between consecutive chunks for continuity.
    """
    long_text = " ".join([f"w{i}" for i in range(400)])
    page = _page(1, [(long_text, 11)])

    chunks = chunk_pages([page], max_tokens=80, overlap=20)

    # Consecutive chunks should share some common tokens
    assert len(chunks) >= 2
    overlap = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert overlap


def test_empty_page_yields_no_chunks():
    """
    Verify that a page with no content produces no chunks.
    """
    page = _page(1, [])
    assert chunk_pages([page], max_tokens=100, overlap=10) == []


def test_bbox_preserved_as_union():
    """
    Verify that the bounding box of a chunk correctly covers all its constituent blocks.
    The chunk bbox should be the union of block bboxes.
    """
    # Two blocks at different vertical positions
    page = _page(1, [("a", 11), ("b", 11)])

    chunks = chunk_pages([page], max_tokens=1000, overlap=0)

    assert len(chunks) == 1
    x0, y0, x1, y1 = chunks[0].bbox
    # The union should cover both blocks
    assert y0 == 0
    assert y1 == 20


def test_ocr_pages_mark_chunk_source_at_creation():
    """
    Verify that OCR-designated pages produce chunks tagged with source="ocr"
    at creation time, while other pages remain source="pdf_text".
    """
    page1 = _page(1, [("digital text", 11)])
    page2 = _page(2, [("ocr text", 11)])

    chunks = chunk_pages([page1, page2], max_tokens=100, overlap=0, ocr_pages={2})

    sources_by_page = {chunk.page: chunk.source for chunk in chunks}
    assert sources_by_page[1] == "pdf_text"
    assert sources_by_page[2] == "ocr"


def test_section_title_carries_forward_across_pages():
    """
    A heading on one page should become the retrieval section title for
    following body-only pages until another heading overrides it.
    """
    page1 = _page(1, [("INSURANCE", 24), ("coverage terms", 11)])
    page2 = _page(2, [("foregoing provisions apply to subcontractors", 11)])

    chunks = chunk_pages([page1, page2], max_tokens=100, overlap=0)

    assert [getattr(chunk, "section_title", None) for chunk in chunks] == [
        None,
        "INSURANCE",
    ]


def test_last_heading_wins_for_subsection_chunks():
    """
    The most recent detected heading should override an earlier one even when
    it uses a smaller font, as long as it still counts as a heading.
    """
    page1 = _page(
        1,
        [
            ("INSURANCE", 24),
            ("general body", 11),
            ("6.B Subcontractor Coverage", 18),
            ("subcontractor body", 11),
        ],
    )
    page2 = _page(2, [("continued subcontractor body", 11)])

    chunks = chunk_pages([page1, page2], max_tokens=100, overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "INSURANCE general body",
        "6.B Subcontractor Coverage subcontractor body",
        "continued subcontractor body",
    ]
    assert [getattr(chunk, "section_title", None) for chunk in chunks] == [
        None,
        "INSURANCE",
        "6.B Subcontractor Coverage",
    ]


def test_indexed_text_prefixes_section_title_only_when_present():
    """
    The retrieval-only indexed text should prepend the section title when one
    exists, while leaving plain chunks untouched.
    """
    titled = chunker.Chunk(
        page=1,
        ordinal=0,
        text="foregoing provisions",
        token_count=2,
        bbox=(0, 0, 10, 10),
        source="pdf_text",
        section_title="INSURANCE",
    )
    untitled = chunker.Chunk(
        page=1,
        ordinal=1,
        text="plain text",
        token_count=2,
        bbox=(0, 0, 10, 10),
        source="pdf_text",
        section_title=None,
    )

    assert chunker.indexed_text(titled) == "INSURANCE\n\nforegoing provisions"
    assert chunker.indexed_text(untitled) == "plain text"
    assert (
        chunker.indexed_text_from_parts("foregoing provisions", "INSURANCE")
        == "INSURANCE\n\nforegoing provisions"
    )
