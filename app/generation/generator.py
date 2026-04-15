"""Generation helpers for cited answers."""

from __future__ import annotations

from typing import Literal

from app.generation.templates import (
    build_json_prompt,
    build_list_prompt,
    build_prose_prompt,
    build_table_prompt,
)
from app.generation.verifier import parse_citation_tags, split_answer_sentences
from app.mistral_client import MistralProtocol

Format = Literal["prose", "list", "table", "json"]


def split_sentences(text: str) -> list[str]:
    """Backwards-compatible wrapper around the shared sentence splitter."""
    return split_answer_sentences(text)


def extract_citations(text: str) -> list[list[int]]:
    """Backwards-compatible wrapper around the shared citation parser."""
    return parse_citation_tags(text)


def generate_answer(
    client: MistralProtocol,
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> str:
    """
    Generate an answer to a query using retrieved context chunks.

    This function builds a prompt from the query and chunks, sends it to the
    LLM client, and returns the generated text.

    Args:
        client: The LLM client following the MistralProtocol.
        query: The user's original search query.
        chunks: A list of (index, text) tuples representing retrieved context.
        disclaimer: An optional key to look up a disclaimer to include in the prompt.

    Returns:
        The generated answer as a string.
    """
    answer, _ = generate_shaped_answer(
        client,
        query=query,
        chunks=chunks,
        format="prose",
        disclaimer=disclaimer,
    )
    return answer


def generate_shaped_answer(
    client: MistralProtocol,
    *,
    query: str,
    chunks: list[tuple[int, str]],
    format: Format,
    disclaimer: str | None,
) -> tuple[str, dict | None]:
    if format == "prose":
        messages = build_prose_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
        response = client.chat(messages)
        if isinstance(response, dict):
            return str(response.get("text", "")), None
        return str(response), None

    if format == "list":
        messages = build_list_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
        response = client.chat(messages, response_format={"type": "json_object"})
        return _parse_structured_response(response, key="items")

    if format == "table":
        messages = build_table_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
        response = client.chat(messages, response_format={"type": "json_object"})
        return _parse_structured_response(response, key="rows")

    messages = build_json_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
    response = client.chat(messages, response_format={"type": "json_object"})
    return _parse_structured_response(response, key="structured")


def _parse_structured_response(response: object, *, key: str) -> tuple[str, dict | None]:
    if not isinstance(response, dict):
        return str(response), None

    answer = response.get("answer")
    structured = response.get(key)
    if isinstance(answer, str) and isinstance(structured, (dict, list)):
        return answer, {key: structured} if key in {"items", "rows"} else structured

    if isinstance(answer, str):
        return answer, None

    return str(response), None
