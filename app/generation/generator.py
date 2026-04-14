"""
Core generation logic for the StackAI RAG application.

This module handles the interaction with the LLM to generate answers based on
retrieved context chunks, as well as post-processing of the generated text
to identify and extract citations.
"""

from __future__ import annotations

import re

from app.generation.templates import build_prose_prompt
from app.mistral_client import MistralProtocol

# Regex for splitting text into sentences, looking for typical sentence endings followed by whitespace and a capital letter.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Regex for identifying citations in the format [n], where n is a number.
CITE_RE = re.compile(r"\[(\d+)\]")


def split_sentences(text: str) -> list[str]:
    """
    Split a block of text into individual sentences.

    Args:
        text: The raw text string to split.

    Returns:
        A list of trimmed sentence strings.
    """
    text = text.strip()
    if not text:
        return []
    # Split using the predefined regex and filter out any empty results.
    return [sentence.strip() for sentence in SENT_SPLIT.split(text) if sentence.strip()]


def extract_citations(text: str) -> list[list[int]]:
    """
    Extract citation numbers from each sentence in the provided text.

    Args:
        text: The generated text containing potential [n] citations.

    Returns:
        A list of lists, where each inner list contains the integer citation
        indices found in the corresponding sentence.
    """
    out: list[list[int]] = []
    # Process each sentence individually to maintain the sentence-to-citation mapping.
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
    # Construct the messages list for the chat completion.
    messages = build_prose_prompt(query=query, chunks=chunks, disclaimer=disclaimer)
    response = client.chat(messages)
    
    # Handle different response formats from the client.
    if isinstance(response, dict):
        return str(response.get("text", ""))
    return str(response)
