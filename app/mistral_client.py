"""Mistral AI client wrapper for embedding, chat, and OCR services.

This module provides an abstraction over the Mistral AI SDK, ensuring
consistent behavior for document processing and retrieval tasks.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class MistralProtocol(Protocol):
    """Protocol defining the interface for the Mistral AI client."""

    def embed(self, text: str) -> np.ndarray:
        """Embeds a single string into a vector."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embeds a batch of strings into a matrix."""
        ...

    def chat(self, messages: list[dict], response_format: Any = None) -> Any:
        """Performs a chat completion request."""
        ...

    def ocr(self, pdf_bytes: bytes) -> str:
        """Extracts markdown-formatted text from a PDF file."""
        ...


class MistralClient:
    """Thin wrapper around the mistralai SDK.

    This client handles low-level details like API communication, vector
    normalization, and batch processing for embeddings.
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        chat_model: str,
        rerank_model: str,
        ocr_model: str,
        embedding_dim: int,
    ):
        """Initializes the Mistral AI client with the provided settings."""
        from mistralai.client import Mistral

        self._client = Mistral(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.rerank_model = rerank_model
        self.ocr_model = ocr_model
        self.embedding_dim = embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Convenience method to embed a single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embeds a batch of strings and returns a normalized NumPy matrix.

        The resulting vectors are normalized to unit length (L2 norm) so that
        dot products can be used as a measure of cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self.embedding_model, inputs=texts)
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        matrix = np.stack(vectors)

        # Normalize to unit length for cosine similarity calculation
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return (matrix / norms).astype(np.float32)

    def chat(self, messages: list[dict], response_format: Any = None) -> Any:
        """Sends a chat request to the Mistral AI API.

        Supports structured JSON output if a response_format is provided.
        """
        import json

        kwargs: dict[str, Any] = {"model": self.chat_model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.complete(**kwargs)
        content = response.choices[0].message.content
        if response_format is not None:
            return json.loads(content)
        return content

    def ocr(self, pdf_bytes: bytes) -> str:
        """Extracts text content from a PDF using Mistral's OCR service.

        Returns page-delimited markdown text. Pages are separated with form-feed
        characters so callers can preserve page boundaries when needed.
        """
        from base64 import b64encode

        # PDF content must be base64-encoded for the process endpoint
        encoded = b64encode(pdf_bytes).decode("ascii")
        response = self._client.ocr.process(
            model=self.ocr_model,
            document={
                "type": "document_base64",
                "document_base64": encoded,
                "document_name": "document.pdf",
            },
        )
        return "\f".join(page.markdown or "" for page in response.pages)


def get_mistral_client() -> MistralProtocol:
    """Factory function for creating the default MistralClient."""
    from app.config import get_settings

    settings = get_settings()
    return MistralClient(
        api_key=settings.mistral_api_key,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        rerank_model=settings.rerank_model,
        ocr_model=settings.ocr_model,
        embedding_dim=settings.embedding_dim,
    )
