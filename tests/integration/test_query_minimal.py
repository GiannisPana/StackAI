"""
Integration tests for the RAG query FastAPI endpoint.

These tests verify the full "Retrieval-Augmented Generation" flow:
1. Document ingestion into the system.
2. Querying the /query endpoint.
3. Hybrid retrieval (semantic + keyword).
4. LLM response generation with citations.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.deps import reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient

import app.api.ingest as ingest_mod
import app.api.query as query_mod


def _pdf(text: str) -> bytes:
    """Generate a valid single-page PDF in memory for testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Setup a FastAPI TestClient with a clean environment for query integration tests.

    Mocks both ingestion and query Mistral clients to simulate the full RAG cycle.
    """
    # Environment isolation
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    import app.config as app_config

    # Ensure clean configuration and global state
    app_config._settings = None
    fake = FakeMistralClient(dim=8)

    # Query-transform: classify every query as a SEARCH so retrieval proceeds.
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave",
            "expansion_queries": [],
        },
    )
    # Reranker: return a high score so the threshold gate passes.
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.92}]})
    # Verifier: default to supporting the single cited sentence in happy-path tests.
    fake.register_chat(r"entailment", {"0": True})
    # Generation: return a canned answer that references citation [1].
    fake.register_chat(r"Context:", "Parental leave is sixteen weeks [1].")

    # Inject fake client into both ingest and query modules
    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)
    
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, fake
    
    # Teardown
    reset_store()
    app_config._settings = None


def test_query_happy_path_returns_answer_with_citation(client):
    """
    Verify the complete RAG flow from ingestion to cited answer.

    Tests that a document can be ingested and then immediately queried to
    produce a relevant, cited response.
    """
    # 1. Ingest a document
    c, _ = client
    pdf = _pdf("parental leave is sixteen weeks for all employees")
    c.post("/ingest", files=[("files", ("a.pdf", pdf, "application/pdf"))])

    # 2. Query the system about the ingested content
    response = c.post("/query", json={"query": "how much parental leave?"})

    # 3. Assert the response contains the expected answer and citation
    assert response.status_code == 200
    body = response.json()
    assert "[1]" in body["answer"]
    assert len(body["citations"]) >= 1
    assert body["intent"] == "search"


def test_query_with_no_corpus_returns_refusal(client):
    """
    Verify the system's behavior when no relevant documents exist.

    Ensures the search layer correctly reports 'no results' and the LLM
    provides a structured refusal instead of hallucinating.
    """
    # Query an empty system
    c, _ = client
    response = c.post("/query", json={"query": "what is this?"})

    assert response.status_code == 200
    body = response.json()
    # The system should identify that it lacks evidence to answer the query
    assert body["refusal_reason"] == "insufficient_evidence"
    assert body["citations"] == []


def test_query_document_ids_filters_results_and_warns_on_missing(client):
    """document_ids must restrict citations to the requested ready documents."""
    c, _ = client
    first_pdf = _pdf("benefits guide: parental leave is sixteen weeks")
    second_pdf = _pdf("engineering handbook unrelated content")

    first = c.post("/ingest", files=[("files", ("first.pdf", first_pdf, "application/pdf"))]).json()["ingested"][0]
    c.post("/ingest", files=[("files", ("second.pdf", second_pdf, "application/pdf"))]).json()["ingested"][0]

    response = c.post(
        "/query",
        json={
            "query": "how much parental leave?",
            "document_ids": [first["document_id"], 99999],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["warnings"]
    assert "99999" in body["warnings"][0]
    assert body["citations"]
    assert {citation["document_id"] for citation in body["citations"]} == {first["document_id"]}


def test_query_verification_flags_uncited_sentences(client):
    """Verification metadata must flag answer sentences that carry no citations."""
    c, fake = client
    pdf = _pdf("expense reimbursement requires receipts")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])
    fake._chat_rules.insert(
        0,
        (r"Question: what is the receipt policy\?", "Receipt policy [1]. This sentence has no citation."),
    )

    response = c.post("/query", json={"query": "what is the receipt policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "prose"
    assert body["verification"]["all_supported"] is False
    assert body["verification"]["unsupported_sentences"] == [1]


def test_query_verification_flags_unsupported_cited_sentence(client):
    """A cited sentence must still be flagged when verifier entailment rejects it."""
    c, fake = client
    pdf = _pdf("parental leave is sixteen weeks for all employees")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])
    fake._chat_rules.insert(
        0,
        (r"Question: how much parental leave\?", "Employees receive twenty weeks [1]."),
    )
    fake._chat_rules.insert(0, (r"entailment", {"0": False}))

    response = c.post("/query", json={"query": "how much parental leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["verification"]["all_supported"] is False
    assert body["verification"]["unsupported_sentences"] == [0]


def test_query_legal_topic_prepends_disclaimer_to_answer(client):
    """A legal-topic search response must include the applied disclaimer in the answer text."""
    c, _ = client
    pdf = _pdf("parental leave is sixteen weeks under the handbook")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])

    response = c.post("/query", json={"query": "what does the contract say about parental leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["policy"]["disclaimer"] == "legal"
    assert body["answer"].startswith("Note: this is general information, not legal advice.")


def test_query_list_format_returns_structured_items(client):
    """Explicit list format should return structured items."""
    c, fake = client
    pdf = _pdf("Employees receive sixteen weeks of parental leave.")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])
    fake._chat_rules.insert(
        0,
        (
            r"items",
            {
                "answer": "- Sixteen weeks [1]",
                "items": [{"text": "Sixteen weeks", "citations": [1]}],
            },
        ),
    )

    response = c.post("/query", json={"query": "list the leave benefit", "format": "list"})

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "list"
    assert body["structured"]["items"][0]["citations"] == [1]


def test_query_table_format_returns_structured_rows(client):
    """Explicit table format should return structured rows."""
    c, fake = client
    pdf = _pdf("Parental leave is sixteen weeks. Sick leave is ten days.")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])
    fake._chat_rules.insert(
        0,
        (
            r"rows",
            {
                "answer": "| Policy | Value |\n| --- | --- |\n| Parental leave | 16 weeks [1] |",
                "rows": [{"cells": ["Parental leave", "16 weeks"], "citations": [1]}],
            },
        ),
    )

    response = c.post("/query", json={"query": "compare leave policies", "format": "table"})

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "table"
    assert body["structured"]["rows"][0]["citations"] == [1]


def test_query_json_format_returns_structured_payload(client):
    """Explicit json format should return structured data instead of 422."""
    c, fake = client
    pdf = _pdf("Employees receive sixteen weeks of parental leave.")
    c.post("/ingest", files=[("files", ("policy.pdf", pdf, "application/pdf"))])
    fake._chat_rules.insert(
        0,
        (
            r"structured",
            {
                "answer": "{\"leave_weeks\": 16}",
                "structured": {"leave_weeks": 16},
            },
        ),
    )

    response = c.post("/query", json={"query": "return leave policy as json", "format": "json"})

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "json"
    assert body["structured"] == {"leave_weeks": 16}
