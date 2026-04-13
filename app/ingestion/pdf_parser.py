from __future__ import annotations

from dataclasses import dataclass

import fitz


@dataclass(frozen=True)
class Block:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float


@dataclass(frozen=True)
class PageContent:
    page_num: int
    blocks: list[Block]
    raw_text: str


def parse_pdf(pdf_bytes: bytes) -> list[PageContent]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[PageContent] = []
        for index, page in enumerate(doc):
            blocks: list[Block] = []
            raw_text = page.get_text("text") or ""
            text_dict = page.get_text("dict")
            for raw_block in text_dict.get("blocks", []):
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
    return len(page.raw_text.strip()) < min_chars
