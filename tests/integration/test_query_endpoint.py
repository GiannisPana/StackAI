from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

import app.api.ingest as ingest_mod
import app.api.query as query_mod
import app.config
from app.deps import reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient


def _pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify.*\[SSN\]",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "masked employee identifier",
            "expansion_queries": [],
        },
    )
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "generic employee policy",
            "expansion_queries": [],
        },
    )
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.91}]})
    fake.register_chat(r"Context:", "Records must stay confidential [1].")

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client

    reset_store()
    app.config._settings = None


def test_query_pii_in_input_is_masked_everywhere(client):
    secret = "123-45-6789"
    client.post(
        "/ingest",
        files=[("files", ("records.pdf", _pdf("Employee records must remain confidential."), "application/pdf"))],
    )

    response = client.post("/query", json={"query": f"What does {secret} mean in the employee record?"})

    assert response.status_code == 200
    body = response.json()
    assert body["policy"]["pii_masked"] is True
    assert body["policy"]["pii_entities"] == ["ssn"]
    assert body["debug"]["rewritten_query"] == "masked employee identifier"
    assert secret not in response.text


def test_query_policy_refuses_personalized_legal_advice_before_transform(client):
    response = client.post("/query", json={"query": "Should I sue my employer over this contract dispute?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "refuse"
    assert body["refusal_reason"] == "personalized_advice"
    assert body["citations"] == []
    assert body["policy"]["disclaimer"] == "legal"


# ---------------------------------------------------------------------------
# HyDE integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client_low_rerank(tmp_path, monkeypatch):
    """Client whose rerank score is forced below threshold_low=0.45 to trigger HyDE."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {"intent": "SEARCH", "sub_intent": "factual",
         "rewritten_query": "parental leave", "expansion_queries": []},
    )
    # LOW rerank score (< threshold_low=0.45) → HyDE must fire
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.2}]})
    # HyDE hypothetical answer
    fake.register_chat(
        r"hypothetical|retrieval probe",
        "Parental leave entitles employees to time off after the birth of a child.",
    )
    # Generation and verifier
    fake.register_chat(r"Context:", "Parental leave is available [1].")
    fake.register_chat(r"entailed|supported", {"1": True})

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        c.post(
            "/ingest",
            files=[("files", ("hr.pdf",
                              _pdf("Parental leave entitles employees to twelve weeks off."),
                              "application/pdf"))],
        )
        yield c, fake
    reset_store()
    app.config._settings = None


@pytest.fixture
def client_high_rerank(tmp_path, monkeypatch):
    """Client whose rerank score is high (>= threshold_low) so HyDE must NOT fire."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    app.config._settings = None

    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {"intent": "SEARCH", "sub_intent": "factual",
         "rewritten_query": "parental leave", "expansion_queries": []},
    )
    # HIGH rerank score → HyDE must NOT fire
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.9}]})
    fake.register_chat(r"Context:", "Parental leave is available [1].")
    fake.register_chat(r"entailed|supported", {"1": True})

    monkeypatch.setattr(ingest_mod, "get_mistral_client", lambda: fake)
    monkeypatch.setattr(query_mod, "get_mistral_client", lambda: fake)

    application = create_app()
    with TestClient(application) as c:
        c.post(
            "/ingest",
            files=[("files", ("hr.pdf",
                              _pdf("Parental leave entitles employees to twelve weeks off."),
                              "application/pdf"))],
        )
        yield c, fake
    reset_store()
    app.config._settings = None


def test_hyde_fires_when_rerank_score_below_threshold(client_low_rerank):
    """used_hyde must be True in debug when top rerank score < threshold_low."""
    c, _ = client_low_rerank
    r = c.post("/query", json={"query": "parental leave?", "enable_llm_rerank": True})
    assert r.status_code == 200
    body = r.json()
    assert body["debug"]["used_hyde"] is True


def test_hyde_skipped_when_rerank_score_high(client_high_rerank):
    """used_hyde must be False when rerank score is above threshold_low."""
    c, fake = client_high_rerank
    calls_before = fake.chat_call_count
    r = c.post("/query", json={"query": "parental leave?", "enable_llm_rerank": True})
    assert r.status_code == 200
    body = r.json()
    assert body["debug"]["used_hyde"] is False
    # Sanity: HyDE would add an extra chat call; assert it didn't
    # transform(1) + rerank(1) + generate(1) + verify(1) = 4 calls
    calls_after = fake.chat_call_count - calls_before
    assert calls_after <= 4, f"Expected ≤4 chat calls (no HyDE), got {calls_after}"
