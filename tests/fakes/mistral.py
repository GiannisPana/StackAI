"""
Fake Mistral API client for deterministic testing.

This module provides a mock implementation of the Mistral API client that
generates stable, reproducible results without making actual network calls.
Embeddings are generated deterministically based on input text hashing,
and chat/OCR responses can be pre-registered for specific inputs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

DEFAULT_CHAT_TEXT = "Default LLM response."
DEFAULT_CHAT_JSON = {"ok": True}


class FakeMistralClient:
    """
    Simulates Mistral API behavior with deterministic outputs.

    This client allows tests to run without an internet connection and ensures
    that the same input always produces the same output (embeddings) or
    allows explicit control over responses (chat, OCR).
    """

    def __init__(self, dim: int = 1024):
        """
        Initialize the fake client.

        Args:
            dim: The dimensionality of the embedding vectors to generate.
        """
        self.dim = dim
        self._fixed_vectors: dict[str, np.ndarray] = {}
        self._chat_rules: list[tuple[str, Any]] = []
        self._ocr_rules: dict[bytes, str] = {}
        self._ocr_page_rules: dict[int, str] = {}
        self.embed_call_count = 0
        self.chat_call_count = 0

    def register_vector(self, text: str, vector: np.ndarray) -> None:
        """
        Manually associate a specific vector with a text string.

        This is useful for testing specific retrieval scenarios where
        precise vector relationships are needed.
        """
        value = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(value)
        if norm == 0:
            raise ValueError("zero vector")
        self._fixed_vectors[text] = value / norm

    def register_chat(self, pattern: str, response: Any) -> None:
        """
        Register a regex pattern to trigger a specific chat response.

        When the chat content matches the pattern, the provided response
        is returned instead of the default.
        """
        self._chat_rules.append((pattern, response))

    def register_ocr(self, pdf_bytes: bytes, text: str) -> None:
        """
        Register specific OCR output for a given set of PDF bytes.
        """
        self._ocr_rules[pdf_bytes] = text

    def register_ocr_by_page_count(self, page_count: int, text: str) -> None:
        """
        Register OCR output for any PDF with the given number of pages.

        Useful when the exact PDF bytes are non-deterministic (e.g. when
        apply_ocr_fallback generates per-page PDFs internally via fitz, which
        embeds a random document ID).  Exact-bytes rules take priority.
        """
        self._ocr_page_rules[page_count] = text

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding for the given text.

        If a vector was manually registered for this text, it is returned.
        Otherwise, a stable pseudo-random vector is generated using the
        SHA-256 hash of the text as a random seed.
        """
        self.embed_call_count += 1
        if text in self._fixed_vectors:
            return self._fixed_vectors[text].copy()

        # Use hash of text as seed for deterministic RNG
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        
        # Generate and normalize the vector
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of strings.
        """
        return np.stack([self.embed(text) for text in texts])

    def chat(self, messages: list[dict], response_format: Any = None) -> Any:
        """
        Simulate a chat completion request.

        Matches the concatenated content of all messages against registered
        regex patterns. Returns the first matching response or a default.
        """
        self.chat_call_count += 1
        content = " ".join(message.get("content", "") for message in messages)
        for pattern, response in self._chat_rules:
            if re.search(pattern, content, re.IGNORECASE):
                return response
        if response_format is not None:
            return dict(DEFAULT_CHAT_JSON)
        return DEFAULT_CHAT_TEXT

    def ocr(self, pdf_bytes: bytes) -> str:
        """
        Simulate an OCR request by looking up registered PDF content.

        Lookup order:
        1. Exact bytes match (``register_ocr``).
        2. Page-count match (``register_ocr_by_page_count``).
        3. Empty string (no match).
        """
        if pdf_bytes in self._ocr_rules:
            return self._ocr_rules[pdf_bytes]
        if self._ocr_page_rules:
            try:
                import fitz as _fitz
                doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
                n = len(doc)
                doc.close()
                if n in self._ocr_page_rules:
                    return self._ocr_page_rules[n]
            except Exception:
                pass
        return ""
