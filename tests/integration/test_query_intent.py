"""
Integration tests for intent-routing in the /query endpoint.

These tests verify that the query transform is wired correctly into the
request-handling pipeline:

* NO_SEARCH queries (e.g. greetings) are short-circuited before any retrieval
  and return a polite acknowledgement with no citations.
* REFUSE queries return a disclaimer without retrieval.
* SEARCH queries use the *rewritten* query for retrieval, not the raw one,
  and expose the rewritten form in the debug block.

All LLM calls are intercepted by :class:`~tests.fakes.mistral.FakeMistralClient`
so no network access is required.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

import app.api.ingest as ingest_mod
import app.api.query as query_mod
import app.config
from app.deps import reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    A TestClient backed by a fully initialised app using a FakeMistralClient.

    The fake is configured with:
    * A classify rule that returns NO_SEARCH when the prompt contains "hello".
    * A generic classify rule that returns SEARCH with "parental leave" as the
      rewritten query for all other classify calls.
    * A generation rule that returns a canned answer when "Context:" appears.

    DEBUG mode is enabled so that debug.rewritten_query is present in responses.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")

    # Reset cached settings so the monkeypatched env vars take effect.
    app.config._settings = None

    fake = FakeMistralClient(dim=8)

    # Specific patterns must be registered before the generic catch-all so that
    # the first-match rule in FakeMistralClient picks the right response.

    # Greeting → NO_SEARCH.
    fake.register_chat(
        r"classify.*hello",
        {
            "intent": "NO_SEARCH",
            "sub_intent": None,
            "rewritten_query": "",
            "expansion_queries": [],
        },
    )
    # Personal legal advice → REFUSE.
    fake.register_chat(
        r"classify.*sue",
        {
            "intent": "REFUSE",
            "sub_intent": None,
            "rewritten_query": "",
            "expansion_queries": [],
        },
    )
    # Generic catch-all: any other classify call → SEARCH.
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave",
            "expansion_queries": [],
        },
    )
    # Generation response for any prompt that contains retrieved context.
    fake.register_chat(r"Context:", "Sixteen weeks [1].")
    # Verifier response for any entailment batch.
    fake.register_chat(r"entailment", {"0": True})

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        yield c

    reset_store()
    app.config._settings = None


# ---------------------------------------------------------------------------
# NO_SEARCH short-circuit
# ---------------------------------------------------------------------------


def test_greeting_skips_retrieval(client):
    """A greeting must return no_search intent and an empty citations list."""
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "no_search"
    assert body["citations"] == []


def test_greeting_returns_200_with_non_empty_answer(client):
    """The short-circuit answer must not be empty."""
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 200
    assert r.json()["answer"]


def test_no_search_debug_has_rewritten_query(client):
    """In debug mode the rewritten_query field must be present (empty string)."""
    r = client.post("/query", json={"query": "hello"})
    body = r.json()
    assert "debug" in body and body["debug"] is not None
    assert body["debug"]["rewritten_query"] == ""


# ---------------------------------------------------------------------------
# REFUSE short-circuit
# ---------------------------------------------------------------------------


def test_refuse_intent_returned(client):
    """Requests for personal legal / medical advice must be refused."""
    r = client.post("/query", json={"query": "should I sue my employer?"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "refuse"
    assert body["citations"] == []
    assert body["refusal_reason"] is not None
    assert body["policy"]["disclaimer"] == "legal"


# ---------------------------------------------------------------------------
# SEARCH path — rewritten query used for retrieval
# ---------------------------------------------------------------------------


def test_search_uses_rewritten_query(client):
    """
    The /query endpoint must forward the *rewritten* query to hybrid_retrieve
    and expose it in debug.rewritten_query.
    """
    pdf = _make_pdf("parental leave is sixteen weeks")
    client.post(
        "/ingest",
        files=[("files", ("hr.pdf", pdf, "application/pdf"))],
    )
    r = client.post(
        "/query",
        json={"query": "how much time off for new parents?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "search"
    assert body["debug"]["rewritten_query"] == "parental leave"


def test_search_intent_present_in_response(client):
    """A normal factual query must return intent == 'search'."""
    r = client.post("/query", json={"query": "what are the leave policies?"})
    assert r.status_code == 200
    assert r.json()["intent"] == "search"


def test_search_debug_includes_verify_latency(client):
    """Debug timings should expose the verifier stage once M9 is wired."""
    pdf = _make_pdf("parental leave is sixteen weeks")
    client.post(
        "/ingest",
        files=[("files", ("hr.pdf", pdf, "application/pdf"))],
    )
    r = client.post(
        "/query",
        json={"query": "how much time off for new parents?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "verify" in body["debug"]["latency_ms"]


def test_auto_format_routes_list_and_comparison_subintents(client):
    """format=auto should map list -> list and comparison -> table."""
    pdf = _make_pdf("parental leave is sixteen weeks. sick leave is ten days.")
    client.post(
        "/ingest",
        files=[("files", ("hr.pdf", pdf, "application/pdf"))],
    )

    query_mod.get_mistral_client()._chat_rules.insert(  # type: ignore[attr-defined]
        0,
        (
            r"classify.*list",
            {
                "intent": "SEARCH",
                "sub_intent": "list",
                "rewritten_query": "leave benefits",
                "expansion_queries": [],
            },
        ),
    )
    query_mod.get_mistral_client()._chat_rules.insert(  # type: ignore[attr-defined]
        0,
        (
            r"classify.*compare",
            {
                "intent": "SEARCH",
                "sub_intent": "comparison",
                "rewritten_query": "leave comparison",
                "expansion_queries": [],
            },
        ),
    )
    query_mod.get_mistral_client()._chat_rules.insert(  # type: ignore[attr-defined]
        0,
        (
            r"items",
            {
                "answer": "- Parental leave [1]",
                "items": [{"text": "Parental leave", "citations": [1]}],
            },
        ),
    )
    query_mod.get_mistral_client()._chat_rules.insert(  # type: ignore[attr-defined]
        0,
        (
            r"rows",
            {
                "answer": "| Policy | Value |\n| --- | --- |\n| Parental leave | 16 weeks [1] |",
                "rows": [{"cells": ["Parental leave", "16 weeks"], "citations": [1]}],
            },
        ),
    )

    list_response = client.post("/query", json={"query": "list the leave benefits", "format": "auto"})
    table_response = client.post("/query", json={"query": "compare leave policies", "format": "auto"})

    assert list_response.status_code == 200
    assert table_response.status_code == 200
    assert list_response.json()["format"] == "list"
    assert table_response.json()["format"] == "table"
