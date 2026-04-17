"""
Unit tests for the text embedding logic using the Mistral client.

These tests verify that texts are correctly batch-embedded and that the
resulting vectors are properly normalized for cosine similarity.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.embeddings import embed_texts
from tests.fakes.mistral import FakeMistralClient


def test_embed_batch_returns_normalized_matrix():
    """
    Verify that embed_texts returns a correctly shaped and L2-normalized matrix.
    L2 normalization is critical for using inner product as a cosine similarity proxy.
    """
    fake = FakeMistralClient(dim=8)

    # Embed three texts
    matrix = embed_texts(fake, ["a", "b", "c"])

    # Result should be (num_texts, embedding_dim)
    assert matrix.shape == (3, 8)
    
    # Each row should be a unit vector (norm = 1.0)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_empty_returns_zero_row_matrix():
    """
    Verify that embedding an empty list of texts returns an appropriately shaped empty matrix.
    """
    fake = FakeMistralClient(dim=8)

    matrix = embed_texts(fake, [])

    # Should have 0 rows but preserve the embedding dimension
    assert matrix.shape == (0, 8)
