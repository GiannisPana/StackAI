"""
Templates and prompt building logic for the StackAI RAG application.

This module defines the system prompts and formatting logic used to construct
inputs for the LLM. It includes pre-defined disclaimers for specific domains.
"""

from __future__ import annotations

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


def _format_chunks(chunks: list[tuple[int, str]]) -> str:
    """
    Format a list of context chunks into a single numbered string.

    Args:
        chunks: A list of (index, text) tuples.

    Returns:
        A formatted string with chunks separated by double newlines.
    """
    return "\n\n".join(f"[{index}] {text}" for index, text in chunks)


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
    # Start with the core context and question.
    parts = [f"Context:\n{_format_chunks(chunks)}", f"Question: {query}"]
    
    # Append a disclaimer instruction if a valid disclaimer key was provided.
    if disclaimer and disclaimer in DISCLAIMERS:
        parts.append(f"Prepend this disclaimer to your answer: {DISCLAIMERS[disclaimer]}")
    
    # Return the formatted messages following the standard OpenAI/Mistral format.
    return [
        {"role": "system", "content": SYSTEM_PROSE},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
