"""Generation helpers for cited answers."""

from __future__ import annotations

from typing import Literal

from app.generation.templates import build_prompt
from app.mistral_client import MistralProtocol

Format = Literal["prose", "list", "table", "json"]
_STRUCTURED_KEYS: dict[Format, str | None] = {
    "prose": None,
    "list": "items",
    "table": "rows",
    "json": "structured",
}


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
    messages, response_format = build_prompt(
        format=format,
        query=query,
        chunks=chunks,
        disclaimer=disclaimer,
    )
    response = client.chat(messages, response_format=response_format)
    structured_key = _STRUCTURED_KEYS[format]
    if structured_key is None:
        if isinstance(response, dict):
            return str(response.get("text", "")), None
        return str(response), None
    return _parse_structured_response(response, key=structured_key)


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
