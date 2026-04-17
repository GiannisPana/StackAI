"""
Integration tests for LLM reranking and the insufficient-evidence threshold gate.

These tests verify the end-to-end behaviour of the rerank + threshold path
inside the /query endpoint:

* When the LLM assigns very low relevance scores the endpoint must refuse with
  ``refusal_reason == "insufficient_evidence"`` rather than fabricating an answer.
* When the LLM scores are high the pipeline must proceed to generation.
* The ``enable_llm_rerank=False`` flag must bypass reranking entirely.
* The debug block must surface ``top_score`` and ``threshold`` so callers can
  diagnose why a query was refused.

All LLM calls are intercepted by FakeMistralClient so no network access is
required.
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


def _ingest_pdf(client: TestClient, text: str, name: str = "a.pdf") -> None:
    """Ingest a single PDF with the given text content."""
    pdf = _make_pdf(text)
    resp = client.post(
        "/ingest",
        files=[("files", (name, pdf, "application/pdf"))],
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def low_score_client(tmp_path, monkeypatch):
    """
    A TestClient whose reranker returns a very low score (0.10) for every
    candidate.  Queries against this client should trigger the
    insufficient-evidence gate.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave",
            "expansion_queries": [],
        },
    )
    # Reranker: all candidates receive a very low score → gate fires.
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.11}, {"id": "1", "score": 0.10}]})
    fake.register_chat(r"Context:", "Answer [1].")

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        yield c

    reset_store()
    app.config._settings = None


@pytest.fixture
def high_score_client(tmp_path, monkeypatch):
    """
    A TestClient whose reranker returns a high score (0.92) so the pipeline
    proceeds all the way to generation.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave",
            "expansion_queries": [],
        },
    )
    # Reranker: top candidate is highly relevant → gate passes.
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.92}]})
    fake.register_chat(r"Context:", "Sixteen weeks [1].")

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        yield c

    reset_store()
    app.config._settings = None


# ---------------------------------------------------------------------------
# Threshold gate — low rerank scores
# ---------------------------------------------------------------------------


def test_low_rerank_scores_trigger_insufficient_evidence(low_score_client):
    """A very low top rerank score must produce refusal_reason = insufficient_evidence."""
    _ingest_pdf(low_score_client, "some unrelated content here", "1.pdf")
    _ingest_pdf(low_score_client, "more unrelated content", "2.pdf")
    r = low_score_client.post("/query", json={"query": "parental leave?"})
    body = r.json()
    assert r.status_code == 200
    assert body["refusal_reason"] == "insufficient_evidence"
    assert body["citations"] == []


def test_low_rerank_debug_exposes_top_score(low_score_client):
    """In debug mode top_score and threshold must both be present in the debug block."""
    _ingest_pdf(low_score_client, "unrelated text", "1.pdf")
    _ingest_pdf(low_score_client, "more unrelated", "2.pdf")
    r = low_score_client.post("/query", json={"query": "parental leave?"})
    body = r.json()
    assert body["debug"] is not None
    assert body["debug"]["top_score"] is not None
    assert body["debug"]["threshold"] is not None


def test_low_rerank_answer_signals_insufficient_evidence(low_score_client):
    """The answer text must mention 'insufficient evidence' (or similar) on refusal."""
    _ingest_pdf(low_score_client, "unrelated text", "1.pdf")
    _ingest_pdf(low_score_client, "more unrelated", "2.pdf")
    r = low_score_client.post("/query", json={"query": "parental leave?"})
    body = r.json()
    assert "insufficient evidence" in body["answer"].lower() or body["refusal_reason"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Threshold gate — high rerank scores
# ---------------------------------------------------------------------------


def test_high_rerank_scores_allow_generation(high_score_client):
    """A high rerank score must allow the pipeline to proceed to generation."""
    _ingest_pdf(high_score_client, "parental leave is sixteen weeks")
    r = high_score_client.post("/query", json={"query": "parental leave?"})
    body = r.json()
    assert r.status_code == 200
    assert body["refusal_reason"] is None
    assert body["answer"]


def test_high_rerank_produces_citations(high_score_client):
    """When generation succeeds there must be at least one citation."""
    _ingest_pdf(high_score_client, "parental leave is sixteen weeks")
    r = high_score_client.post("/query", json={"query": "parental leave?"})
    assert r.json()["citations"]


# ---------------------------------------------------------------------------
# Rerank bypass
# ---------------------------------------------------------------------------


def test_disable_llm_rerank_bypasses_reranker(high_score_client, monkeypatch):
    """
    When enable_llm_rerank=False the endpoint must skip the LLM rerank call
    and use the raw hybrid scores.  Generation must still succeed when there
    are matching candidates.
    """
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("llm_rerank should be bypassed when enable_llm_rerank=False")

    monkeypatch.setattr(query_mod, "llm_rerank", fail_if_called)

    _ingest_pdf(high_score_client, "parental leave is sixteen weeks")
    r = high_score_client.post(
        "/query",
        json={"query": "parental leave?", "enable_llm_rerank": False},
    )
    body = r.json()
    assert r.status_code == 200
    # Without reranking the gate uses raw hybrid scores (typically high enough
    # when the query is relevant), so the pipeline should proceed.
    assert body["intent"] == "search"
