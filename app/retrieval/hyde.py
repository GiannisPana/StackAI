"""Adaptive HyDE (Hypothetical Document Embeddings) fallback for weak retrievals.

HyDE fires after LLM rerank when the top rerank score is below threshold_low.
The LLM generates a concise hypothetical answer paragraph that is then embedded
and used as an additional retrieval vector, improving recall on ambiguous queries.

Gate semantics: compare the normalised LLM rerank score (0..1 scale) against
threshold_low, NOT the pre-fusion RRF score. RRF rank-1 scores are ~0.016 with
k=60 and would always trigger HyDE. Rerank scores are meaningfully calibrated.
"""
from __future__ import annotations

from app.mistral_client import MistralProtocol

_HYDE_PROMPT = (
    "Write a short, factual hypothetical answer paragraph (3-5 sentences) "
    "to the user's question, as if you were quoting from a reference document. "
    "Do not hedge. Do not say 'I don't know'. The text will be used only as a "
    "retrieval probe to find real source documents.\n\nQuestion: {query}"
)


def hyde_expand(client: MistralProtocol, query: str) -> str:
    """Ask the LLM for a hypothetical answer paragraph to use as a retrieval probe.

    Args:
        client: Mistral client for chat completions.
        query: The user's search query.

    Returns:
        A short hypothetical answer string, stripped of leading/trailing whitespace.
        Returns an empty string if the LLM response is empty.
    """
    messages = [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}]
    response = client.chat(messages)
    return (response or "").strip()
