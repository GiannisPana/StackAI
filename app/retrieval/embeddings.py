"""Utilities for generating text embeddings.

This module provides functions to interface with embedding models, handling
batching and dimensionality details.
"""

from __future__ import annotations

import numpy as np

from app.mistral_client import MistralProtocol


def embed_texts(client: MistralProtocol, texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using the provided client.

    Args:
        client: The Mistral API client or a compatible protocol.
        texts: A list of strings to embed.

    Returns:
        A NumPy array of shape (num_texts, embedding_dim) containing the embeddings.
        Returns an empty array with the correct dimensions if 'texts' is empty.
    """
    if not texts:
        # Determine embedding dimension from client attributes if available
        dim = getattr(client, "dim", None) or getattr(client, "embedding_dim", 0)
        return np.zeros((0, int(dim)), dtype=np.float32)
    return client.embed_batch(texts)
