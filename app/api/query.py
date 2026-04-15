"""
Query endpoints for the StackAI RAG application.

This module implements the full RAG loop in eight stages:

1. Policy application — PII masking, disclaimer detection, and refusal.
2. Query transform — intent classification and retrieval rewriting via LLM.
3. Short-circuit for NO_SEARCH (greetings / chit-chat) and REFUSE (policy).
4. Hybrid retrieval (vector + BM25 + RRF) using the rewritten query.
5. LLM rerank over the top-20 candidates + insufficient-evidence threshold gate.
6. MMR diversity selection over the reranked set.
7. Generation of a cited answer using the configured LLM.
8. Citation-anchored verification of the final answer.

The threshold gate (step 4) implements the bonus "citations required" feature:
if the top rerank score is below ``settings.threshold_high`` *and* the score
distribution is insufficiently spread (no clear winner), the endpoint refuses
with ``refusal_reason="insufficient_evidence"`` rather than hallucinating.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import Citation, DebugInfo, PolicyInfo, QueryRequest, QueryResponse
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
        if spread >= spread_min and spread > 0:
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
        return QueryResponse(
            answer=_apply_disclaimer(
                "I can't provide personal advice on this topic. "
                "I can summarise what the uploaded documents say about it if you'd like.",
                policy.disclaimer,
            ),
            format=response_format,
            intent="refuse",
            refusal_reason=policy.refuse_reason,
            citations=[],
            warnings=warnings,
            policy=policy_info,
            debug=DebugInfo(
                latency_ms={"total": int((time.monotonic() - t0) * 1000)},
            )
            if settings.debug
            else None,
        )

    # ------------------------------------------------------------------
    # Step 2: Classify intent and rewrite query for retrieval.
    # ------------------------------------------------------------------
    transform = transform_query(client, masked_query)
    timings["transform"] = int((time.monotonic() - t0) * 1000)
    response_format = _resolve_output_format(req.format, transform.sub_intent)

    # Short-circuit: greetings and chit-chat — no retrieval needed.
    if transform.intent == "no_search":
        return QueryResponse(
            answer=_apply_disclaimer(
                "Hello! Ask me anything about your uploaded documents.",
                policy.disclaimer,
            ),
            format=response_format,
            intent="no_search",
            sub_intent=transform.sub_intent,
            warnings=warnings,
            policy=policy_info,
            debug=DebugInfo(
                rewritten_query=transform.rewritten_query,
                latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
            )
            if settings.debug
            else None,
        )

    # Short-circuit: personalised legal / medical advice — policy refusal.
    if transform.intent == "refuse":
        return QueryResponse(
            answer=_apply_disclaimer(
                "I can't provide personal advice on this topic. "
                "I can summarise what the uploaded documents say about it if you'd like."
            , policy.disclaimer),
            format=response_format,
            intent="refuse",
            sub_intent=transform.sub_intent,
            refusal_reason=policy.refuse_reason or "personalized_advice",
            warnings=warnings,
            policy=policy_info,
            debug=DebugInfo(
                rewritten_query=transform.rewritten_query,
                latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
            )
            if settings.debug
            else None,
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
        return QueryResponse(
            answer=_apply_disclaimer(
                "I don't have sufficient evidence in the indexed documents to answer this question.",
                policy.disclaimer,
            ),
            format=response_format,
            intent="search",
            sub_intent=transform.sub_intent,
            refusal_reason="insufficient_evidence",
            citations=[],
            warnings=warnings,
            policy=policy_info,
            debug=DebugInfo(
                rewritten_query=transform.rewritten_query,
                expansion_queries=transform.expansion_queries,
                num_candidates=0,
                threshold=settings.threshold_high,
                latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
            )
            if settings.debug
            else None,
        )

    # ------------------------------------------------------------------
    # Step 5: LLM rerank over the top-N window + threshold gate.
    # ------------------------------------------------------------------
    rerank_window: list[Candidate] = candidates[:_RERANK_WINDOW]

    # Pre-load chunk texts needed for the rerank prompt.
    rerank_rows = [c.row for c in rerank_window]
    chunks_by_row = _load_chunks(rerank_rows)
    chunk_texts = {r: chunks_by_row[r]["text"] for r in rerank_rows if r in chunks_by_row}

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
    used_hyde = False
    if (
        req.enable_llm_rerank
        and rerank_window
        and rerank_window[0].score < settings.threshold_low
    ):
        hyde_start = time.monotonic()
        hypothetical = hyde_expand(client, retrieval_query)
        if hypothetical:
            hyde_vec = client.embed(hypothetical)
            candidates = hybrid_retrieve(
                client,
                query=retrieval_query,
                mask=mask,
                active_rows=active_rows,
                expansion_queries=transform.expansion_queries or None,
                extra_vectors=[hyde_vec],
            )
            # Re-run rerank on the refreshed candidate set
            rerank_window = candidates[:_RERANK_WINDOW]
            rerank_rows = [c.row for c in rerank_window]
            chunks_by_row = _load_chunks(rerank_rows)
            chunk_texts = {r: chunks_by_row[r]["text"] for r in rerank_rows if r in chunks_by_row}
            if chunk_texts:
                rerank_window = llm_rerank(
                    client,
                    query=retrieval_query,
                    candidates=rerank_window,
                    chunk_texts=chunk_texts,
                )
            used_hyde = True
        timings["hyde"] = int((time.monotonic() - hyde_start) * 1000)

    # Threshold gate — refuse if evidence quality is too low.
    passed, top_score = _threshold_passed(
        rerank_window,
        threshold_high=settings.threshold_high,
        spread_min=settings.spread_min,
    )
    if not passed:
        return QueryResponse(
            answer=_apply_disclaimer(
                "I don't have sufficient evidence in the indexed documents to answer this question.",
                policy.disclaimer,
            ),
            format=response_format,
            intent="search",
            sub_intent=transform.sub_intent,
            refusal_reason="insufficient_evidence",
            citations=[],
            warnings=warnings,
            policy=policy_info,
            debug=DebugInfo(
                rewritten_query=transform.rewritten_query,
                expansion_queries=transform.expansion_queries,
                used_hyde=used_hyde,
                num_candidates=len(candidates),
                top_score=float(top_score),
                threshold=settings.threshold_high,
                latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
            )
            if settings.debug
            else None,
        )

    # ------------------------------------------------------------------
    # Step 6: MMR diversity selection over the reranked set.
    # ------------------------------------------------------------------
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
    chunks_by_row = _load_chunks(final_rows)

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
    answer = _apply_disclaimer(answer, policy.disclaimer)
    chunk_lookup = {index: text for index, text in numbered}
    verify_start = time.monotonic()
    verification = verify_answer(client, answer=answer, chunk_lookup=chunk_lookup)
    timings["verify"] = int((time.monotonic() - verify_start) * 1000)

    # ------------------------------------------------------------------
    # Step 8: Map citations into the response payload.
    # ------------------------------------------------------------------
    citations: list[Citation] = []
    for index, row in enumerate(final_rows, start=1):
        chunk = chunks_by_row.get(row)
        if chunk is None:
            continue
        score = next(c.score for c in top if c.row == row)
        citations.append(
            Citation(
                index=index,
                document_id=int(chunk["doc_id"]),
                filename=str(chunk["filename"]),
                page=int(chunk["page"]),
                chunk_id=int(chunk["id"]),
                text=str(chunk["text"]),
                score=float(score),
            )
        )

    return QueryResponse(
        answer=answer,
        format=response_format,
        intent="search",
        sub_intent=transform.sub_intent,
        citations=citations,
        structured=structured,
        verification=verification,
        policy=policy_info,
        warnings=warnings,
        debug=DebugInfo(
            rewritten_query=transform.rewritten_query,
            expansion_queries=transform.expansion_queries,
            used_hyde=used_hyde,
            num_candidates=len(candidates),
            top_score=float(top[0].score),
            threshold=settings.threshold_high,
            latency_ms=timings | {"total": int((time.monotonic() - t0) * 1000)},
        )
        if settings.debug
        else None,
    )
