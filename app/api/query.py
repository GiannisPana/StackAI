"""
Query endpoints for the StackAI RAG application.

This module implements the full RAG loop in eight stages:

1. Policy application — PII masking, disclaimer detection, and refusal.
2. Query transform — intent classification and retrieval rewriting via LLM.
3. Short-circuit for NO_SEARCH (greetings / chit-chat) and REFUSE (policy).
4. Hybrid retrieval (vector + BM25 + RRF) using the rewritten query.
5. LLM rerank over the top-20 candidates; HyDE fallback re-retrieval + re-rerank
   if the top rerank score is below ``threshold_low``; insufficient-evidence
   threshold gate.
6. MMR diversity selection over the reranked set.
7. Generation of a cited answer using the configured LLM.
8. Citation-anchored verification of the final answer.

The threshold gate (step 5) implements the bonus "citations required" feature:
if the top rerank score is below ``settings.threshold_high`` *and* the score
distribution is insufficiently spread (no clear winner), the endpoint refuses
with ``refusal_reason="insufficient_evidence"`` rather than hallucinating.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import Citation, DebugInfo, PolicyInfo, QueryRequest, QueryResponse, Verification
from app.config import get_settings
from app.deps import get_store
from app.generation.generator import generate_shaped_answer
from app.generation.policy import apply_policy
from app.generation.query_transform import transform_query
from app.generation.templates import DISCLAIMERS
from app.generation.verifier import verify_answer
from app.mistral_client import get_mistral_client
from app.retrieval.hyde import hyde_expand
from app.retrieval.mmr import mmr_select
from app.retrieval.rerank import llm_rerank
from app.retrieval.search import Candidate, hybrid_retrieve
from app.storage.db import get_connection
from app.storage.repository import resolve_document_filter, row_set_for_documents

router = APIRouter()

_RERANK_WINDOW = 20


def _load_chunks(rows: list[int]) -> dict[int, dict]:
    """
    Load chunk metadata from the database for the given embedding row indices.

    Args:
        rows: A list of integer row indices in the embedding matrix.

    Returns:
        A dictionary mapping embedding_row to a dictionary of chunk attributes
        with keys ``id``, ``embedding_row``, ``text``, ``page``, ``doc_id``,
        and ``filename``.
    """
    if not rows:
        return {}
    # Use standard SQL placeholders for the IN clause.
    placeholders = ",".join("?" * len(rows))
    conn = get_connection()
    try:
        results = conn.execute(
            f"SELECT c.id, c.embedding_row, c.text, c.page, c.doc_id, d.filename "
            f"FROM chunks c JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.embedding_row IN ({placeholders})",
            rows,
        ).fetchall()
    finally:
        conn.close()
    return {int(row["embedding_row"]): dict(row) for row in results}


def _load_chunk_texts(rows: list[int]) -> tuple[dict[int, dict], dict[int, str]]:
    """Load chunk metadata and the plain-text lookup used by rerank prompts."""
    chunks_by_row = _load_chunks(rows)
    chunk_texts = {row: str(chunks_by_row[row]["text"]) for row in rows if row in chunks_by_row}
    return chunks_by_row, chunk_texts


def _threshold_passed(
    candidates: list[Candidate],
    *,
    threshold_high: float,
    spread_min: float,
) -> tuple[bool, float]:
    """
    Decide whether the top candidate clears the evidence threshold.

    The gate passes when *either* condition holds:
    - The top score exceeds ``threshold_high`` (clearly relevant).
    - The score distribution has a spread >= ``spread_min`` between the best
      and median candidate, indicating a clear winner even if absolute scores
      are moderate.

    Args:
        candidates: Reranked candidates, sorted descending by score.
        threshold_high: Minimum score for unconditional acceptance.
        spread_min: Minimum (top − median) / top ratio for spread-based acceptance.

    Returns:
        ``(passed, top_score)`` — whether the gate cleared and the best score.
    """
    if not candidates:
        return False, 0.0

    top_score = candidates[0].score
    if top_score >= threshold_high:
        return True, top_score

    # Plan: single candidate pools below threshold_high are OK
    if len(candidates) < 2:
        return True, top_score

    # Spread check: a clear winner can pass even below threshold_high.
    median = candidates[len(candidates) // 2].score
    if top_score > 0:
        spread = (top_score - median) / top_score
        if spread >= spread_min:
            return True, top_score

    return False, top_score


def _apply_disclaimer(answer: str, disclaimer: str | None) -> str:
    """Prepend the configured disclaimer text if one should be applied."""
    if not disclaimer or disclaimer not in DISCLAIMERS:
        return answer

    prefix = DISCLAIMERS[disclaimer]
    text = answer.strip()
    if text.startswith(prefix):
        return text
    if not text:
        return prefix
    return f"{prefix}\n\n{text}"


def _resolve_output_format(requested: str, sub_intent: str | None) -> str:
    if requested != "auto":
        return requested
    if sub_intent == "list":
        return "list"
    if sub_intent == "comparison":
        return "table"
    return "prose"


def _build_response(
    *,
    answer: str,
    intent: str,
    policy_info: PolicyInfo,
    warnings: list[str],
    timings: dict[str, int],
    t0: float,
    settings,
    format: str = "prose",
    sub_intent: str | None = None,
    refusal_reason: str | None = None,
    citations: list[Citation] | None = None,
    structured: dict | None = None,
    verification: Verification | None = None,
    debug_extra: dict | None = None,
) -> QueryResponse:
    """Build a consistent QueryResponse for both short-circuit and success paths."""
    debug = None
    if settings.debug:
        debug = DebugInfo(
            latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
            **(debug_extra or {}),
        )

    payload = {
        "answer": _apply_disclaimer(answer, policy_info.disclaimer),
        "format": format,
        "intent": intent,
        "sub_intent": sub_intent,
        "refusal_reason": refusal_reason,
        "citations": citations or [],
        "policy": policy_info,
        "warnings": warnings,
        "debug": debug,
    }
    if structured is not None:
        payload["structured"] = structured
    if verification is not None:
        payload["verification"] = verification
    return QueryResponse(**payload)


def _maybe_hyde_rerank(
    client,
    *,
    retrieval_query: str,
    candidates: list[Candidate],
    rerank_window: list[Candidate],
    mask: set[int] | None,
    active_rows: set[int],
    expansion_queries: list[str] | None,
    enable_rerank: bool,
    threshold_low: float,
    timings: dict[str, int],
) -> tuple[list[Candidate], list[Candidate], bool]:
    """Retry retrieval with a HyDE vector when the top rerank score is weak."""
    if not (enable_rerank and rerank_window and rerank_window[0].score < threshold_low):
        return candidates, rerank_window, False

    hyde_start = time.monotonic()
    refreshed_candidates = candidates
    refreshed_window = rerank_window
    used_hyde = False

    hypothetical = hyde_expand(client, retrieval_query)
    if hypothetical:
        hyde_vec = client.embed(hypothetical)
        refreshed_candidates = hybrid_retrieve(
            client,
            query=retrieval_query,
            mask=mask,
            active_rows=active_rows,
            expansion_queries=expansion_queries,
            extra_vectors=[hyde_vec],
        )
        refreshed_window = refreshed_candidates[:_RERANK_WINDOW]
        _, chunk_texts = _load_chunk_texts([candidate.row for candidate in refreshed_window])
        if chunk_texts:
            refreshed_window = llm_rerank(
                client,
                query=retrieval_query,
                candidates=refreshed_window,
                chunk_texts=chunk_texts,
            )
        used_hyde = True

    timings["hyde"] = int((time.monotonic() - hyde_start) * 1000)
    return refreshed_candidates, refreshed_window, used_hyde


def _build_citations(
    *,
    final_rows: list[int],
    chunks_by_row: dict[int, dict],
    candidates: list[Candidate],
) -> list[Citation]:
    """Map the final ranked rows into the API citation payload."""
    scores_by_row = {candidate.row: candidate.score for candidate in candidates}
    citations: list[Citation] = []
    for index, row in enumerate(final_rows, start=1):
        chunk = chunks_by_row.get(row)
        if chunk is None:
            continue
        citations.append(
            Citation(
                index=index,
                document_id=int(chunk["doc_id"]),
                filename=str(chunk["filename"]),
                page=int(chunk["page"]),
                chunk_id=int(chunk["id"]),
                text=str(chunk["text"]),
                score=float(scores_by_row[row]),
            )
        )
    return citations


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Execute a RAG query against the indexed documents.

    Runs the full pipeline: intent transform → retrieval → rerank + threshold →
    MMR → generation → citation mapping.

    Args:
        req: A QueryRequest object containing the query and options.

    Returns:
        A QueryResponse containing the generated answer, citations, and metadata.
        If insufficient evidence is found the answer is replaced with a refusal.
    """
    settings = get_settings()
    client = get_mistral_client()
    store = get_store()
    
    # Snapshot ready_rows for consistency throughout the request
    active_rows = store.ready_rows
    
    t0 = time.monotonic()
    timings: dict[str, int] = {}
    warnings: list[str] = []
    response_format = "prose"

    # ------------------------------------------------------------------
    # Step 1: Apply policy before any other processing.
    # ------------------------------------------------------------------
    policy = apply_policy(client, req.query)
    masked_query = policy.masked_text
    policy_info = PolicyInfo(
        pii_masked=bool(policy.entities),
        pii_entities=policy.entities,
        disclaimer=policy.disclaimer,
    )

    if policy.refuse_reason is not None:
        return _build_response(
            answer=(
                "I can't provide personal advice on this topic. "
                "I can summarise what the uploaded documents say about it if you'd like."
            ),
            intent="refuse",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            refusal_reason=policy.refuse_reason,
        )

    # ------------------------------------------------------------------
    # Step 2: Classify intent and rewrite query for retrieval.
    # ------------------------------------------------------------------
    transform = transform_query(client, masked_query)
    timings["transform"] = int((time.monotonic() - t0) * 1000)
    response_format = _resolve_output_format(req.format, transform.sub_intent)

    # Short-circuit: greetings and chit-chat — no retrieval needed.
    if transform.intent == "no_search":
        return _build_response(
            answer="Hello! Ask me anything about your uploaded documents.",
            intent="no_search",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            sub_intent=transform.sub_intent,
            debug_extra={
                "rewritten_query": transform.rewritten_query,
            },
        )

    # Short-circuit: personalised legal / medical advice — policy refusal.
    if transform.intent == "refuse":
        return _build_response(
            answer=(
                "I can't provide personal advice on this topic. "
                "I can summarise what the uploaded documents say about it if you'd like."
            ),
            intent="refuse",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            sub_intent=transform.sub_intent,
            refusal_reason=policy.refuse_reason or "personalized_advice",
            debug_extra={
                "rewritten_query": transform.rewritten_query,
            },
        )

    # ------------------------------------------------------------------
    # Step 3: Handle document filtering if document_ids are provided.
    # ------------------------------------------------------------------
    mask: set[int] | None = None
    if req.document_ids is not None:
        conn = get_connection()
        try:
            # Check which document IDs are valid and ready.
            valid, missing = resolve_document_filter(conn, req.document_ids)
            if not valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "no_valid_documents",
                        "requested": req.document_ids,
                        "missing": missing,
                    },
                )
            if missing:
                warnings.append(
                    f"document_ids {missing} not found or not ready; searched only {valid}"
                )
            # Create a mask of row indices for the allowed documents.
            mask = row_set_for_documents(conn, valid)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Step 4: Hybrid retrieval (vector + BM25 + RRF).
    # ------------------------------------------------------------------
    retrieval_query = transform.rewritten_query or masked_query
    retrieve_start = time.monotonic()
    candidates = hybrid_retrieve(
        client,
        query=retrieval_query,
        mask=mask,
        active_rows=active_rows,
        expansion_queries=transform.expansion_queries or None,
    )
    timings["retrieve"] = int((time.monotonic() - retrieve_start) * 1000)

    if not candidates:
        return _build_response(
            answer="I don't have sufficient evidence in the indexed documents to answer this question.",
            intent="search",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            sub_intent=transform.sub_intent,
            refusal_reason="insufficient_evidence",
            debug_extra={
                "rewritten_query": transform.rewritten_query,
                "expansion_queries": transform.expansion_queries,
                "num_candidates": 0,
                "threshold": settings.threshold_high,
            },
        )

    # ------------------------------------------------------------------
    # Step 5: LLM rerank over the top-N window + threshold gate.
    # ------------------------------------------------------------------
    rerank_window: list[Candidate] = candidates[:_RERANK_WINDOW]

    # Pre-load chunk texts needed for the rerank prompt.
    rerank_rows = [c.row for c in rerank_window]
    _, chunk_texts = _load_chunk_texts(rerank_rows)

    if req.enable_llm_rerank and chunk_texts:
        rerank_start = time.monotonic()
        rerank_window = llm_rerank(
            client,
            query=retrieval_query,
            candidates=rerank_window,
            chunk_texts=chunk_texts,
        )
        timings["rerank"] = int((time.monotonic() - rerank_start) * 1000)

    # ------------------------------------------------------------------
    # HyDE fallback — fires post-rerank when top rerank score is weak.
    # Uses normalised rerank scores (0..1), not pre-fusion RRF scores.
    # ------------------------------------------------------------------
    candidates, rerank_window, used_hyde = _maybe_hyde_rerank(
        client,
        retrieval_query=retrieval_query,
        candidates=candidates,
        rerank_window=rerank_window,
        mask=mask,
        active_rows=active_rows,
        expansion_queries=transform.expansion_queries or None,
        enable_rerank=req.enable_llm_rerank,
        threshold_low=settings.threshold_low,
        timings=timings,
    )

    # Threshold gate — refuse if evidence quality is too low.
    passed, top_score = _threshold_passed(
        rerank_window,
        threshold_high=settings.threshold_high,
        spread_min=settings.spread_min,
    )
    if not passed:
        return _build_response(
            answer="I don't have sufficient evidence in the indexed documents to answer this question.",
            intent="search",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            sub_intent=transform.sub_intent,
            refusal_reason="insufficient_evidence",
            debug_extra={
                "rewritten_query": transform.rewritten_query,
                "expansion_queries": transform.expansion_queries,
                "used_hyde": used_hyde,
                "num_candidates": len(candidates),
                "top_score": float(top_score),
                "threshold": settings.threshold_high,
            },
        )

    # ------------------------------------------------------------------
    # Step 6: MMR diversity selection over the reranked set.
    # ------------------------------------------------------------------
    # Drop zero-scored candidates (reranker omitted them from its JSON response)
    # only here, after the threshold gate, so HyDE still sees the full window.
    rerank_window = [c for c in rerank_window if c.score > 0.0]
    # If the filter empties the window, the reranker scored no candidate as
    # relevant — treat as insufficient evidence. This also covers the
    # single-candidate edge case where the threshold gate bypass lets a
    # score=0.0 row through.
    if not rerank_window:
        return _build_response(
            answer="I don't have sufficient evidence in the indexed documents to answer this question.",
            intent="search",
            policy_info=policy_info,
            warnings=warnings,
            timings=timings,
            t0=t0,
            settings=settings,
            format=response_format,
            sub_intent=transform.sub_intent,
            refusal_reason="insufficient_evidence",
            debug_extra={
                "rewritten_query": transform.rewritten_query,
                "expansion_queries": transform.expansion_queries,
                "used_hyde": used_hyde,
                "num_candidates": len(candidates),
                "top_score": float(top_score),
                "threshold": settings.threshold_high,
            },
        )
    if len(rerank_window) > req.top_k:
        store = get_store()
        rerank_vecs = np.stack([store.embeddings[c.row] for c in rerank_window])
        relevance_scores = [c.score for c in rerank_window]
        picked_indices = mmr_select(
            vectors=rerank_vecs,
            relevance=relevance_scores,
            k=req.top_k,
            lambda_=settings.mmr_lambda,
        )
        top = [rerank_window[i] for i in picked_indices]
    else:
        top = rerank_window[: req.top_k]

    # Refetch chunk metadata for the final top set (may differ from rerank window).
    final_rows = [c.row for c in top]
    chunks_by_row, _ = _load_chunk_texts(final_rows)

    # ------------------------------------------------------------------
    # Step 7: Generate the answer.
    # ------------------------------------------------------------------
    numbered = [
        (index + 1, chunks_by_row[row]["text"])
        for index, row in enumerate(final_rows)
        if row in chunks_by_row
    ]

    generate_start = time.monotonic()
    answer, structured = generate_shaped_answer(
        client,
        query=masked_query,
        chunks=numbered,
        format=response_format,
        disclaimer=policy.disclaimer,
    )
    timings["generate"] = int((time.monotonic() - generate_start) * 1000)
    # Apply disclaimer before verification so sentence indexes in
    # verification.unsupported_sentences match the string the client receives.
    answer = _apply_disclaimer(answer, policy.disclaimer)
    chunk_lookup = {index: text for index, text in numbered}
    verify_start = time.monotonic()
    verification = verify_answer(client, answer=answer, chunk_lookup=chunk_lookup)
    timings["verify"] = int((time.monotonic() - verify_start) * 1000)

    # ------------------------------------------------------------------
    # Step 8: Map citations into the response payload.
    # ------------------------------------------------------------------
    citations = _build_citations(
        final_rows=final_rows,
        chunks_by_row=chunks_by_row,
        candidates=top,
    )

    return _build_response(
        answer=answer,
        intent="search",
        policy_info=policy_info,
        warnings=warnings,
        timings=timings,
        t0=t0,
        settings=settings,
        format=response_format,
        sub_intent=transform.sub_intent,
        citations=citations,
        structured=structured,
        verification=verification,
        debug_extra={
            "rewritten_query": transform.rewritten_query,
            "expansion_queries": transform.expansion_queries,
            "used_hyde": used_hyde,
            "num_candidates": len(candidates),
            "top_score": float(top[0].score),
            "threshold": settings.threshold_high,
        },
    )
