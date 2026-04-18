from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.deps import Store, get_store, reset_store, set_store
from app.generation.query_transform import transform_query
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.bm25 import tokenize
from app.retrieval.hyde import hyde_expand
from app.retrieval.rerank import llm_rerank
from app.retrieval.search import hybrid_retrieve
from app.storage.bm25_store import load_bm25
from app.storage.db import get_connection, init_schema
from tests.fakes.mistral import FakeMistralClient


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "SampleContract-Shuttle.pdf"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    import app.config

    app.config._settings = None
    init_schema()
    settings = get_settings()

    store = Store(embeddings=np.zeros((0, settings.embedding_dim), dtype=np.float32))
    store.bm25 = load_bm25(settings.bm25_path)
    set_store(store)

    yield

    reset_store()
    app.config._settings = None


def test_section_title_makes_page_five_retrievable(env):
    fake = FakeMistralClient(dim=8)
    pdf_bytes = FIXTURE.read_bytes()

    result = ingest_pdf(fake, filename="SampleContract-Shuttle.pdf", pdf_bytes=pdf_bytes)

    assert result["status"] == "ready"

    store = get_store()
    query = "What are the insurance flow-down requirements for subcontractors?"
    candidates = hybrid_retrieve(
        fake,
        query=query,
        mask=None,
        active_rows=store.ready_rows,
    )
    candidate_rows = [candidate.row for candidate in candidates]

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT embedding_row, page FROM chunks WHERE embedding_row IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    page_by_row = {int(row["embedding_row"]): int(row["page"]) for row in rows}
    assert any(page_by_row.get(row) == 5 for row in candidate_rows)

    bm25_hits = dict(store.bm25.top_k(tokenize(query), k=50, mask=store.ready_rows))
    assert any(page_by_row.get(row) == 5 for row in bm25_hits)


def test_retrieval_path_is_deterministic_across_identical_runs(env):
    fake = FakeMistralClient(dim=8)
    fake.register_chat(
        r"classify",
        {
            "intent": "SEARCH",
            "sub_intent": "factual",
            "rewritten_query": "parental leave policy",
            "expansion_queries": [],
        },
    )
    fake.register_chat(r"relevance", {"scores": [{"id": "0", "score": 0.92}]})
    fake.register_chat(
        r"hypothetical answer paragraph",
        "Parental leave is sixteen weeks.",
    )

    pdf_bytes = (
        b"%PDF-1.4\n"
    )
    # Use the real ingestion path so hybrid retrieval operates on live stores.
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Parental leave is sixteen weeks long.", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = ingest_pdf(fake, filename="policy.pdf", pdf_bytes=pdf_bytes)
    assert result["status"] == "ready"

    store = get_store()

    def run_once():
        transform = transform_query(fake, "How much parental leave is available?")
        candidates = hybrid_retrieve(
            fake,
            query=transform.rewritten_query,
            mask=None,
            active_rows=store.ready_rows,
        )
        rows = [candidate.row for candidate in candidates[:1]]
        chunk_texts = {rows[0]: "Parental leave is sixteen weeks long."}
        reranked = llm_rerank(
            fake,
            query=transform.rewritten_query,
            candidates=candidates[:1],
            chunk_texts=chunk_texts,
        )
        hypothetical = hyde_expand(fake, transform.rewritten_query)
        hyde_candidates = hybrid_retrieve(
            fake,
            query=transform.rewritten_query,
            mask=None,
            active_rows=store.ready_rows,
            extra_vectors=[fake.embed(hypothetical)],
        )
        hyde_reranked = llm_rerank(
            fake,
            query=transform.rewritten_query,
            candidates=hyde_candidates[:1],
            chunk_texts=chunk_texts,
        )
        return {
            "rewritten_query": transform.rewritten_query,
            "reranked_rows": [candidate.row for candidate in reranked],
            "hyde_text": hypothetical,
            "hyde_reranked_rows": [candidate.row for candidate in hyde_reranked],
        }

    before = len(fake.chat_calls)
    first = run_once()
    first_calls = fake.chat_calls[before:]

    before = len(fake.chat_calls)
    second = run_once()
    second_calls = fake.chat_calls[before:]

    assert first == second
    assert all(call["temperature"] == 0.0 for call in first_calls)
    assert all(call["temperature"] == 0.0 for call in second_calls)
    assert len(first_calls) == len(second_calls)
    assert len(first_calls) >= 4
