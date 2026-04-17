from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

import app.api.query as query_mod
from app.api.schemas import Citation, PolicyInfo
from app.retrieval.search import Candidate


def _citation() -> Citation:
    return Citation(
        index=1,
        document_id=7,
        filename="policy.pdf",
        page=2,
        chunk_id=11,
        text="Policy excerpt",
        score=0.91,
    )


def test_build_response_applies_disclaimer_and_debug_metadata():
    settings = SimpleNamespace(debug=True)
    policy_info = PolicyInfo(
        pii_masked=True,
        pii_entities=["email"],
        disclaimer="legal",
    )

    response = query_mod._build_response(
        answer="Contract summary [1].",
        intent="search",
        policy_info=policy_info,
        warnings=["missing secondary document"],
        timings={"transform": 5},
        t0=time.monotonic() - 0.01,
        format="prose",
        sub_intent="factual",
        citations=[_citation()],
        settings=settings,
        debug_extra={
            "rewritten_query": "contract summary",
            "threshold": 0.55,
        },
    )

    assert response.answer.startswith("Note: this is general information, not legal advice.")
    assert response.policy == policy_info
    assert response.citations[0].filename == "policy.pdf"
    assert response.debug is not None
    assert response.debug.rewritten_query == "contract summary"
    assert response.debug.threshold == 0.55
    assert response.debug.latency_ms["transform"] == 5
    assert "total" in response.debug.latency_ms


def test_build_response_omits_debug_when_disabled():
    settings = SimpleNamespace(debug=False)
    response = query_mod._build_response(
        answer="Hello!",
        intent="no_search",
        policy_info=PolicyInfo(),
        warnings=[],
        timings={},
        t0=time.monotonic(),
        format="prose",
        settings=settings,
    )

    assert response.answer == "Hello!"
    assert response.debug is None
    assert response.intent == "no_search"


def test_load_chunk_texts_returns_chunk_lookup_and_plain_texts(monkeypatch):
    monkeypatch.setattr(
        query_mod,
        "_load_chunks",
        lambda rows: {
            1: {"text": "alpha"},
            3: {"text": "gamma"},
        },
    )

    chunks_by_row, chunk_texts = query_mod._load_chunk_texts([1, 2, 3])

    assert chunks_by_row[1]["text"] == "alpha"
    assert chunk_texts == {
        1: "alpha",
        3: "gamma",
    }


def test_maybe_hyde_rerank_refreshes_candidates_when_top_score_is_weak(monkeypatch):
    initial_candidates = [Candidate(row=1, score=0.2)]
    refreshed_candidates = [Candidate(row=2, score=0.8)]
    reranked_candidates = [Candidate(row=2, score=0.91)]
    timings: dict[str, int] = {}

    class DummyClient:
        def embed(self, text: str) -> np.ndarray:
            assert text == "hypothetical answer"
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(query_mod, "hyde_expand", lambda client, query: "hypothetical answer")
    monkeypatch.setattr(query_mod, "hybrid_retrieve", lambda client, **kwargs: refreshed_candidates)
    monkeypatch.setattr(
        query_mod,
        "_load_chunk_texts",
        lambda rows: ({2: {"text": "supporting chunk"}}, {2: "supporting chunk"}),
    )
    monkeypatch.setattr(
        query_mod,
        "llm_rerank",
        lambda client, query, candidates, chunk_texts: reranked_candidates,
    )

    refreshed, reranked, used_hyde = query_mod._maybe_hyde_rerank(
        DummyClient(),
        retrieval_query="leave policy",
        candidates=initial_candidates,
        rerank_window=initial_candidates,
        mask=None,
        active_rows={2},
        expansion_queries=["leave policy benefits"],
        enable_rerank=True,
        threshold_low=0.45,
        timings=timings,
    )

    assert refreshed == refreshed_candidates
    assert reranked == reranked_candidates
    assert used_hyde is True
    assert "hyde" in timings


def test_build_citations_maps_rows_into_response_payload():
    citations = query_mod._build_citations(
        final_rows=[4, 9],
        chunks_by_row={
            4: {
                "id": 101,
                "doc_id": 7,
                "filename": "policy.pdf",
                "page": 2,
                "text": "Parental leave is sixteen weeks.",
            },
            9: {
                "id": 102,
                "doc_id": 8,
                "filename": "benefits.pdf",
                "page": 5,
                "text": "Eligibility begins after six months.",
            },
        },
        candidates=[
            Candidate(row=4, score=0.93),
            Candidate(row=9, score=0.81),
        ],
    )

    assert [citation.index for citation in citations] == [1, 2]
    assert citations[0].filename == "policy.pdf"
    assert citations[1].score == 0.81
