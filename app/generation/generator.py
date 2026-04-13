from __future__ import annotations

import re

from app.generation.templates import build_prose_prompt
from app.mistral_client import MistralProtocol

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
CITE_RE = re.compile(r"\[(\d+)\]")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [sentence.strip() for sentence in SENT_SPLIT.split(text) if sentence.strip()]


def extract_citations(text: str) -> list[list[int]]:
    out: list[list[int]] = []
    for sentence in split_sentences(text):
        out.append([int(match.group(1)) for match in CITE_RE.finditer(sentence)])
    return out


def generate_answer(
    client: MistralProtocol,
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> str:
    messages = build_prose_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
    response = client.chat(messages)
    if isinstance(response, dict):
        return str(response.get("text", ""))
    return str(response)
