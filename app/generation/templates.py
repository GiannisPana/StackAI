"""
Templates and prompt building logic for the StackAI RAG application.

This module defines the system prompts and formatting logic used to construct
inputs for the LLM. It includes pre-defined disclaimers for specific domains.
"""

from __future__ import annotations

from typing import Literal

# Standard disclaimers for specific sensitive domains.
DISCLAIMERS = {
    "legal": (
        "Note: this is general information, not legal advice. Consult a qualified attorney "
        "for advice specific to your situation."
    ),
    "medical": (
        "Note: this is general information, not medical advice. Consult a qualified healthcare "
        "professional."
    ),
}

# The base system prompt that defines the LLM's persona and rules for citation.
SYSTEM_PROSE = (
    "You are a careful assistant answering questions strictly from the provided numbered "
    "context chunks. Cite every sentence that uses information from a chunk by appending "
    "the chunk number in square brackets, e.g. [1] or [1][2]. If the chunks do not contain "
    "enough information to answer, say so briefly instead of guessing."
)

SYSTEM_STRUCTURED = (
    "You are a careful assistant answering strictly from the provided numbered context chunks. "
    "Return STRICT JSON only. Every factual statement in the top-level answer string must keep "
    "inline [n] citations that refer to the provided chunk numbers."
)


def _format_chunks(chunks: list[tuple[int, str]]) -> str:
    """
    Format a list of context chunks into a single numbered string.

    Args:
        chunks: A list of (index, text) tuples.

    Returns:
        A formatted string with chunks separated by double newlines.
    """
    return "\n\n".join(f"[{index}] {text}" for index, text in chunks)


def _disclaimer_instruction(disclaimer: str | None) -> str | None:
    if disclaimer and disclaimer in DISCLAIMERS:
        return f"Prepend this disclaimer to your answer: {DISCLAIMERS[disclaimer]}"
    return None


def _build_prompt(
    *,
    system_prompt: str,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
    instructions: list[str],
) -> list[dict]:
    parts = [f"Context:\n{_format_chunks(chunks)}", f"Question: {query}", *instructions]
    disclaimer_instruction = _disclaimer_instruction(disclaimer)
    if disclaimer_instruction:
        parts.append(disclaimer_instruction)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_prose_prompt(
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> list[dict]:
    """
    Construct the messages for a prose-based RAG prompt.

    Args:
        query: The user's query.
        chunks: The retrieved context chunks as (index, text) tuples.
        disclaimer: Optional key for a domain-specific disclaimer.

    Returns:
        A list of message dictionaries suitable for a chat completion API.
    """
    return _build_prompt(
        system_prompt=SYSTEM_PROSE,
        query=query,
        chunks=chunks,
        disclaimer=disclaimer,
        instructions=[],
    )


def build_list_prompt(
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> list[dict]:
    return _build_prompt(
        system_prompt=SYSTEM_STRUCTURED,
        query=query,
        chunks=chunks,
        disclaimer=disclaimer,
        instructions=[
            (
                'Return JSON with keys: "answer" and "items". '
                '"answer" must be a bullet list string with inline [n] citations. '
                '"items" must be an array of objects with keys "text" and "citations".'
            )
        ],
    )


def build_table_prompt(
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> list[dict]:
    return _build_prompt(
        system_prompt=SYSTEM_STRUCTURED,
        query=query,
        chunks=chunks,
        disclaimer=disclaimer,
        instructions=[
            (
                'Return JSON with keys: "answer" and "rows". '
                '"answer" must be a markdown table string with inline [n] citations. '
                '"rows" must be an array of objects with keys "cells" and "citations".'
            )
        ],
    )


def build_json_prompt(
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> list[dict]:
    return _build_prompt(
        system_prompt=SYSTEM_STRUCTURED,
        query=query,
        chunks=chunks,
        disclaimer=disclaimer,
        instructions=[
            (
                'Return JSON with keys: "answer" and "structured". '
                '"answer" must be a compact JSON string with inline [n] citations where factual claims appear. '
                '"structured" must be a JSON object representing the answer.'
            )
        ],
    )
