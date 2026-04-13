from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.pdf_parser import Block, PageContent


@dataclass(frozen=True)
class Chunk:
    page: int
    ordinal: int
    text: str
    token_count: int
    bbox: tuple[float, float, float, float]
    source: str


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _is_heading(block: Block, body_size: float) -> bool:
    return block.font_size >= body_size * 1.3


def _union_bbox(blocks: list[Block]) -> tuple[float, float, float, float]:
    return (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )


def _group_by_heading(page: PageContent) -> list[list[Block]]:
    if not page.blocks:
        return []

    sizes = [block.font_size for block in page.blocks if block.font_size > 0]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 11.0
    groups: list[list[Block]] = []
    current: list[Block] = []

    for block in page.blocks:
        if _is_heading(block, body_size) and current:
            groups.append(current)
            current = [block]
        else:
            current.append(block)

    if current:
        groups.append(current)
    return groups


def _split_with_cap(text: str, max_tokens: int, overlap: int) -> list[str]:
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
    chunks: list[Chunk] = []

    for page in pages:
        for group in _group_by_heading(page):
            text = " ".join(block.text for block in group).strip()
            if not text:
                continue
            bbox = _union_bbox(group)
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
