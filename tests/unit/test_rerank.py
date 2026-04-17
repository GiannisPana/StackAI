"""
Unit tests for the LLM-based reranker.

The reranker calls the LLM with a batch of chunk texts and a query, receives
a JSON relevance-score map, and returns candidates sorted by that LLM score.
These tests exercise the happy path, edge cases, and failure modes without
making real network calls.
"""

from __future__ import annotations

import pytest

from app.retrieval.rerank import llm_rerank
from app.retrieval.search import Candidate
from tests.fakes.mistral import FakeMistralClient


def _fake(dim: int = 4) -> FakeMistralClient:
    """Return a fresh fake client."""
    return FakeMistralClient(dim=dim)


def _cands(*rows_scores: tuple[int, float]) -> list[Candidate]:
    """Build a list of Candidates from (row, score) pairs."""
    return [Candidate(row=r, score=s) for r, s in rows_scores]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_llm_scores():
    """Candidates must be re-ordered by the LLM scores, highest first."""
    fake = _fake()
    fake.register_chat(
        r"relevance",
        {
            "scores": [
                {"id": "0", "score": 0.2},
                {"id": "1", "score": 0.95},
                {"id": "2", "score": 0.4},
            ]
        },
    )
    cands = _cands((0, 0.9), (1, 0.5), (2, 0.7))
    chunk_texts = {0: "alpha", 1: "beta", 2: "gamma"}
    out = llm_rerank(fake, query="q", candidates=cands, chunk_texts=chunk_texts)
    assert out[0].row == 1
    assert 0 <= out[0].score <= 1
    assert out[0].score >= out[1].score >= out[2].score


def test_rerank_scores_are_clamped_to_0_1():
    """Scores returned by the LLM outside [0, 1] must be clamped."""
    fake = _fake()
    fake.register_chat(
        r"relevance",
        {
            "scores": [
                {"id": "0", "score": 2.5},   # above 1 → clamped to 1.0
                {"id": "1", "score": -0.3},  # below 0 → clamped to 0.0
            ]
        },
    )
    out = llm_rerank(
        fake,
        query="q",
        candidates=_cands((0, 0.8), (1, 0.6)),
        chunk_texts={0: "a", 1: "b"},
    )
    assert out[0].score == 1.0
    assert out[1].score == 0.0


def test_rerank_preserves_all_candidates():
    """Every input candidate must appear exactly once in the output."""
    fake = _fake()
    fake.register_chat(
        r"relevance",
        {"scores": [{"id": "0", "score": 0.7}, {"id": "1", "score": 0.3}]},
    )
    cands = _cands((10, 0.9), (20, 0.5))
    chunk_texts = {10: "x", 20: "y"}
    out = llm_rerank(fake, query="q", candidates=cands, chunk_texts=chunk_texts)
    assert len(out) == 2
    assert {c.row for c in out} == {10, 20}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rerank_with_empty_candidates_returns_empty():
    """An empty candidate list must produce an empty result immediately."""
    fake = _fake()
    assert llm_rerank(fake, query="q", candidates=[], chunk_texts={}) == []


def test_rerank_missing_chunk_text_uses_empty_string():
    """
    When a row has no entry in chunk_texts the reranker must still include it
    in the prompt (with an empty string) rather than crashing.
    """
    fake = _fake()
    fake.register_chat(
        r"relevance",
        {"scores": [{"id": "0", "score": 0.6}]},
    )
    out = llm_rerank(
        fake,
        query="q",
        candidates=_cands((99, 0.8)),
        chunk_texts={},   # no text for row 99
    )
    assert len(out) == 1
    assert out[0].row == 99


def test_rerank_partial_scores_fallback_to_zero():
    """
    If the LLM omits a score for some candidates the missing ones must default
    to 0.0 rather than being dropped.
    """
    fake = _fake()
    fake.register_chat(
        r"relevance",
        # Only scores candidate 0; candidate 1 is absent.
        {"scores": [{"id": "0", "score": 0.8}]},
    )
    out = llm_rerank(
        fake,
        query="q",
        candidates=_cands((0, 0.9), (1, 0.9)),
        chunk_texts={0: "a", 1: "b"},
    )
    rows = [c.row for c in out]
    assert 0 in rows and 1 in rows
    scored = {c.row: c.score for c in out}
    assert scored[1] == 0.0


# ---------------------------------------------------------------------------
# Failure / robustness
# ---------------------------------------------------------------------------


def test_rerank_malformed_response_keeps_original_order():
    """
    When the LLM returns a response without a 'scores' key the original
    candidate order and scores must be preserved unchanged.
    """
    fake = _fake()  # default returns {"ok": True} — no 'scores' key
    cands = _cands((0, 0.9), (1, 0.5))
    out = llm_rerank(fake, query="q", candidates=cands, chunk_texts={0: "a", 1: "b"})
    assert [c.row for c in out] == [0, 1]
    assert [c.score for c in out] == [0.9, 0.5]


def test_rerank_network_error_keeps_original_order():
    """A client that raises during chat must not propagate the exception."""

    class BrokenClient:
        def chat(self, messages, response_format=None):  # noqa: ANN001
            raise RuntimeError("timeout")

        def embed(self, text):  # noqa: ANN001
            raise NotImplementedError

        def embed_batch(self, texts):  # noqa: ANN001
            raise NotImplementedError

        def ocr(self, pdf_bytes):  # noqa: ANN001
            raise NotImplementedError

    cands = _cands((0, 0.9), (1, 0.5))
    out = llm_rerank(
        BrokenClient(),  # type: ignore[arg-type]
        query="q",
        candidates=cands,
        chunk_texts={0: "a", 1: "b"},
    )
    assert [c.row for c in out] == [0, 1]


def test_rerank_score_entry_with_bad_types_is_skipped():
    """
    Individual score entries with non-numeric / non-integer fields must be
    silently skipped rather than raising.
    """
    fake = _fake()
    fake.register_chat(
        r"relevance",
        {
            "scores": [
                {"id": "not-a-number", "score": 0.9},  # bad id
                {"id": "1", "score": "high"},          # bad score
                {"id": "0", "score": 0.5},             # valid
            ]
        },
    )
    out = llm_rerank(
        fake,
        query="q",
        candidates=_cands((0, 0.8), (1, 0.6)),
        chunk_texts={0: "a", 1: "b"},
    )
    # Only candidate 0 gets a LLM score; candidate 1 defaults to 0.0.
    scored = {c.row: c.score for c in out}
    assert scored[0] == 0.5
    assert scored[1] == 0.0
