"""
Text chunking for PDF documents.

This module provides heuristics for splitting PDF pages into logical chunks
based on font size and structure (layout heuristics) to preserve context
during retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.pdf_parser import Block, PageContent


@dataclass(frozen=True)
class Chunk:
    """
    A logical fragment of text extracted from a PDF.

    Attributes:
        page: 1-based index of the page.
        ordinal: Position of the chunk within the document's sequence.
        text: The actual text content.
        token_count: Estimated number of tokens in the text.
        bbox: (x0, y0, x1, y1) bounding box on the original page.
        source: Information about how the chunk was created.
    """
    page: int
    ordinal: int
    text: str
    token_count: int
    bbox: tuple[float, float, float, float]
    source: str


def _approx_tokens(text: str) -> int:
    """Return an approximate token count for the given text."""
    return len(text.split())


def _is_heading(block: Block, body_size: float) -> bool:
    """
    Determine if a block is a heading based on its font size relative to body text.
    
    Layout heuristic: Headings are typically significantly larger than body text.
    """
    return block.font_size >= body_size * 1.3


def _union_bbox(blocks: list[Block]) -> tuple[float, float, float, float]:
    """Compute the minimal bounding box that contains all given blocks."""
    return (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )


def _group_by_heading(page: PageContent) -> list[list[Block]]:
    """
    Group blocks into sections based on heading detection.
    
    This heuristic assumes that a heading starts a new logical section.
    We identify the body text size by taking the median font size of the page.
    """
    if not page.blocks:
        return []

    # Heuristic: Identify the dominant font size on the page as the body size.
    sizes = [block.font_size for block in page.blocks if block.font_size > 0]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 11.0
    
    groups: list[list[Block]] = []
    current: list[Block] = []

    for block in page.blocks:
        # If we encounter a heading and already have some accumulated blocks,
        # we start a new group.
        if _is_heading(block, body_size) and current:
            groups.append(current)
            current = [block]
        else:
            current.append(block)

    if current:
        groups.append(current)
    return groups


def _split_with_cap(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Split a string into smaller windows of tokens with a specified overlap.
    
    Ensures that no chunk exceeds the maximum token limit.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]

    output: list[str] = []
    step = max(1, max_tokens - overlap)
    index = 0
    while index < len(tokens):
        window = tokens[index : index + max_tokens]
        output.append(" ".join(window))
        if index + max_tokens >= len(tokens):
            break
        index += step
    return output


def chunk_pages(pages: list[PageContent], max_tokens: int, overlap: int) -> list[Chunk]:
    """
    Process a list of pages and return a flat list of chunks.

    Layout heuristics are applied to group related text blocks before
    splitting them into token-capped windows.
    """
    chunks: list[Chunk] = []

    for page in pages:
        # Grouping by heading helps keep related information in the same chunk.
        for group in _group_by_heading(page):
            text = " ".join(block.text for block in group).strip()
            if not text:
                continue
            bbox = _union_bbox(group)
            
            # Apply token limits and overlap to create final chunks.
            for part in _split_with_cap(text, max_tokens=max_tokens, overlap=overlap):
                if not part.strip():
                    continue
                chunks.append(
                    Chunk(
                        page=page.page_num,
                        ordinal=0,
                        text=part,
                        token_count=_approx_tokens(part),
                        bbox=bbox,
                        source="pdf_text",
                    )
                )

    return chunks
