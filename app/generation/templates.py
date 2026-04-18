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
    "the chunk number in square brackets, e.g. [1] or [1][2]. "
    "Use ONLY information present in the provided chunks. Do not rely on general knowledge "
    "of standard industry practices, legal conventions, or typical contract provisions. "
    "If a specific number, threshold, or clause is not present in the chunks, say "
    "'The provided documents do not specify this.' — do not infer it."
)

SYSTEM_STRUCTURED = (
    "You are a careful assistant answering strictly from the provided numbered context chunks. "
    "Return STRICT JSON only. Every factual statement in the top-level answer string must keep "
    "inline [n] citations that refer to the provided chunk numbers. "
    "Use ONLY information present in the provided chunks. Do not rely on general knowledge "
    "of standard industry practices, legal conventions, or typical contract provisions. "
    "If a specific number, threshold, or clause is not present in the chunks, say "
    "'The provided documents do not specify this.' — do not infer it."
)

Format = Literal["prose", "list", "table", "json"]

_FORMAT_SPEC: dict[Format, dict[str, object]] = {
    "prose": {
        "system_prompt": SYSTEM_PROSE,
        "instructions": [],
        "response_format": None,
    },
    "list": {
        "system_prompt": SYSTEM_STRUCTURED,
        "instructions": [
            (
                'Return JSON with keys: "answer" and "items". '
                '"answer" must be a bullet list string with inline [n] citations. '
                '"items" must be an array of objects with keys "text" and "citations".'
            )
        ],
        "response_format": {"type": "json_object"},
    },
    "table": {
        "system_prompt": SYSTEM_STRUCTURED,
        "instructions": [
            (
                'Return JSON with keys: "answer" and "rows". '
                '"answer" must be a markdown table string with inline [n] citations. '
                '"rows" must be an array of objects with keys "cells" and "citations".'
            )
        ],
        "response_format": {"type": "json_object"},
    },
    "json": {
        "system_prompt": SYSTEM_STRUCTURED,
        "instructions": [
            (
                'Return JSON with keys: "answer" and "structured". '
                '"answer" must be a compact JSON string with inline [n] citations where factual claims appear. '
                '"structured" must be a JSON object representing the answer.'
            )
        ],
        "response_format": {"type": "json_object"},
    },
}


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


def build_prompt(
    *,
    format: Format,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> tuple[list[dict], dict | None]:
    """
    Build prompt messages and the optional response_format for a named output shape.

    Supported formats:
    - ``prose``: free-form cited answer
    - ``list``: JSON object with ``answer`` and ``items``
    - ``table``: JSON object with ``answer`` and ``rows``
    - ``json``: JSON object with ``answer`` and ``structured``
    """
    spec = _FORMAT_SPEC[format]
    instructions = list(spec["instructions"])
    parts = [f"Context:\n{_format_chunks(chunks)}", f"Question: {query}", *instructions]
    disclaimer_instruction = _disclaimer_instruction(disclaimer)
    if disclaimer_instruction:
        parts.append(disclaimer_instruction)

    messages = [
        {"role": "system", "content": str(spec["system_prompt"])},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    response_format = spec["response_format"]
    return messages, response_format if isinstance(response_format, dict) else None
