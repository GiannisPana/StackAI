"""
Unit tests for the FakeMistralClient utility.

The FakeMistralClient is used throughout the test suite to simulate Mistral API
calls without making actual network requests. These tests ensure the fake
itself behaves predictably.
"""

from __future__ import annotations

import numpy as np

from tests.fakes.mistral import FakeMistralClient


def test_embed_deterministic():
    """
    Verify that the fake client returns identical, normalized vectors for identical inputs.
    """
    client = FakeMistralClient(dim=8)
    vec1 = client.embed("hello world")
    vec2 = client.embed("hello world")

    # The fake uses a hash-based seed for determinism
    assert np.allclose(vec1, vec2)
    # Vectors should be L2-normalized by default
    assert abs(np.linalg.norm(vec1) - 1.0) < 1e-5


def test_register_vector_overrides():
    """
    Verify the ability to pin specific vectors to specific input texts.
    This is useful for unit testing downstream components like vector search.
    """
    client = FakeMistralClient(dim=4)
    pinned = np.array([1, 0, 0, 0], dtype=np.float32)
    client.register_vector("pinned input", pinned)

    assert np.allclose(client.embed("pinned input"), pinned)


def test_chat_pattern_matching():
    """
    Verify that the fake client can return specific mock responses based on input keywords.
    """
    client = FakeMistralClient(dim=4)
    # Register a response for the keyword "rerank"
    client.register_chat("rerank", {"scores": [{"id": "a", "score": 0.9}]})

    response = client.chat([{"role": "user", "content": "please rerank these"}])

    assert response == {"scores": [{"id": "a", "score": 0.9}]}


def test_chat_default_fallback():
    """
    Verify that text-mode chat falls back to a plain string when no keyword matches.
    """
    client = FakeMistralClient(dim=4)

    response = client.chat([{"role": "user", "content": "unmatched"}])

    assert response == "Default LLM response."


def test_chat_default_fallback_for_structured_output_is_json_like():
    """
    Structured-output calls should fall back to a dict so JSON-parsing callers
    exercise their malformed-response handling instead of type mismatches.
    """
    client = FakeMistralClient(dim=4)

    response = client.chat(
        [{"role": "user", "content": "unmatched"}],
        response_format={"type": "json_object"},
    )

    assert response == {"ok": True}
