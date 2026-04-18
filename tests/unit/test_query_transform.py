"""
Unit tests for the query transform module.

These tests exercise intent classification, query rewriting, and the
fallback behaviour when the LLM returns an unexpected response, all
without making real network calls.
"""

from __future__ import annotations

import pytest

from app.generation.query_transform import QueryTransform, transform_query
from tests.fakes.mistral import FakeMistralClient


def _fake(dim: int = 4) -> FakeMistralClient:
    """Return a fresh fake client with the given embedding dimension."""
    return FakeMistralClient(dim=dim)


# ---------------------------------------------------------------------------
# Intent: NO_SEARCH
# ---------------------------------------------------------------------------


def test_greeting_classified_no_search():
    """Greetings must be classified as NO_SEARCH with an empty rewritten query."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "NO_SEARCH",
            "sub_intent": None,
            "rewritten_query": "",
            "expansion_queries": [],
        },
    )
    result = transform_query(fake, "hello")
    assert result.intent == "no_search"
    assert result.rewritten_query == ""


def test_no_search_returns_empty_expansions():
    """A NO_SEARCH result must always carry an empty expansion list."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "NO_SEARCH",
            "sub_intent": None,
            "rewritten_query": "",
            "expansion_queries": [],
        },
    )
    result = transform_query(fake, "hi there")
    assert result.expansion_queries == []


# ---------------------------------------------------------------------------
# Intent: SEARCH
# ---------------------------------------------------------------------------


def test_search_query_returns_rewritten_and_expansions():
    """A substantive query must be rewritten and expanded for retrieval."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave policy",
            "expansion_queries": ["leave benefits", "time off new parents"],
        },
    )
    result = transform_query(fake, "how much parental leave?")
    assert result.intent == "search"
    assert result.sub_intent == "factual"
    assert result.rewritten_query == "parental leave policy"
    assert len(result.expansion_queries) == 2


def test_expansion_capped_at_three():
    """More than three expansion queries must be silently truncated."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": None,
            "rewritten_query": "topic",
            "expansion_queries": ["a", "b", "c", "d", "e"],
        },
    )
    result = transform_query(fake, "topic")
    assert len(result.expansion_queries) == 3


def test_sub_intent_null_string_normalised():
    """The string literal 'null' returned by the LLM must become Python None."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "null",
            "rewritten_query": "topic",
            "expansion_queries": [],
        },
    )
    result = transform_query(fake, "topic")
    assert result.sub_intent is None


# ---------------------------------------------------------------------------
# Intent: REFUSE
# ---------------------------------------------------------------------------


def test_refuse_intent_preserved():
    """Legal / medical personal-advice queries must yield the REFUSE intent."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "REFUSE",
            "sub_intent": None,
            "rewritten_query": "",
            "expansion_queries": [],
        },
    )
    result = transform_query(fake, "should I sue my employer?")
    assert result.intent == "refuse"


# ---------------------------------------------------------------------------
# Fallback / robustness
# ---------------------------------------------------------------------------


def test_malformed_response_falls_back_to_search_with_raw_query():
    """
    If the LLM returns a response without the expected 'intent' key the
    transform must silently fall back to SEARCH intent, preserving the
    original query as the rewritten form.
    """
    fake = _fake()
    # Default chat returns {"ok": True} which is missing required keys.
    result = transform_query(fake, "anything")
    assert result.intent == "search"
    assert result.rewritten_query == "anything"
    assert result.expansion_queries == []
    assert result.sub_intent is None


def test_network_error_falls_back_gracefully():
    """
    A client that raises during chat must not surface an exception to the caller.
    The transform should fall back to SEARCH with the original query.
    """

    class BrokenClient:
        def chat(self, messages, response_format=None, temperature=None):  # noqa: ANN001
            raise RuntimeError("network down")

        def embed(self, text):  # noqa: ANN001
            raise NotImplementedError

        def embed_batch(self, texts):  # noqa: ANN001
            raise NotImplementedError

        def ocr(self, pdf_bytes):  # noqa: ANN001
            raise NotImplementedError

    result = transform_query(BrokenClient(), "what is the refund policy?")  # type: ignore[arg-type]
    assert result.intent == "search"
    assert result.rewritten_query == "what is the refund policy?"


def test_unknown_intent_string_defaults_to_search():
    """An unrecognised intent value must be treated as SEARCH."""
    fake = _fake()
    fake.register_chat(
        r"classify",
        {
            "intent": "DUNNO",
            "sub_intent": None,
            "rewritten_query": "rewritten",
            "expansion_queries": [],
        },
    )
    result = transform_query(fake, "something")
    assert result.intent == "search"


def test_result_is_frozen_dataclass():
    """QueryTransform instances must be immutable (frozen dataclass)."""
    result = QueryTransform(
        intent="search",
        sub_intent=None,
        rewritten_query="q",
        expansion_queries=[],
    )
    with pytest.raises((AttributeError, TypeError)):
        result.intent = "no_search"  # type: ignore[misc]


def test_transform_query_pins_temperature_to_zero_for_retrieval_calls():
    """Retrieval rewrites should request deterministic chat completions."""

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, messages, response_format=None, temperature=None):  # noqa: ANN001
            self.calls.append(
                {
                    "messages": messages,
                    "response_format": response_format,
                    "temperature": temperature,
                }
            )
            return {
                "intent": "SEARCH",
                "sub_intent": None,
                "rewritten_query": "rewritten",
                "expansion_queries": [],
            }

        def embed(self, text):  # noqa: ANN001
            raise NotImplementedError

        def embed_batch(self, texts):  # noqa: ANN001
            raise NotImplementedError

        def ocr(self, pdf_bytes):  # noqa: ANN001
            raise NotImplementedError

    client = RecordingClient()

    transform_query(client, "original query")  # type: ignore[arg-type]

    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.0
