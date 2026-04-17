"""High-level search orchestration for the RAG pipeline.

This module coordinates multiple retrieval systems (vector search and BM25)
to perform hybrid retrieval, merging results using Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.deps import get_store
from app.mistral_client import MistralProtocol
from app.retrieval.bm25 import tokenize
from app.retrieval.fusion import rrf
from app.retrieval.vector_index import top_k as vector_top_k


@dataclass(frozen=True)
class Candidate:
    """A search result candidate with its document ID and fusion score."""

    row: int
    score: float


def hybrid_retrieve(
    client: MistralProtocol,
    *,
    query: str,
    mask: set[int] | None,
    active_rows: set[int],
    expansion_queries: list[str] | None = None,
    extra_vectors: list[np.ndarray] | None = None,
    per_query_k: int = 50,
) -> list[Candidate]:
    """Perform hybrid retrieval by merging vector search and BM25 results.

    Orchestrates the retrieval pipeline:
    1. Resolve active document mask.
    2. For each query (base + expansions), execute vector search and BM25.
    3. For each extra vector (e.g. HyDE embedding), execute vector search.
    4. Fuse all rankings using Reciprocal Rank Fusion (RRF).

    Args:
        client: API client for generating query embeddings.
        query: Raw search query string.
        mask: Optional set of row IDs to restrict the search.
        active_rows: Set of row IDs for documents ready and not deleted.
        expansion_queries: Additional query strings (e.g. from multi-query expansion).
            Each produces its own cosine + BM25 ranking list fused via RRF.
        extra_vectors: Pre-computed vectors to search against (e.g. HyDE hypothesis
            embedding). Each produces one additional cosine ranking list.
        per_query_k: Number of results to fetch from each retriever.

    Returns:
        A list of hybrid candidates, sorted by RRF score descending.
    """
    store = get_store()
    # `mask` is the caller's scoping filter (e.g. specific doc IDs, or None for "all").
    # `active_rows` is the global "ready & not soft-deleted" set from the store.
    # Effective search set is their intersection.
    effective_mask = active_rows if mask is None else (mask & active_rows)

    all_queries = [query] + list(expansion_queries or [])
    rankings: list[list[tuple[int, float]]] = []

    for q in all_queries:
        # Vector over base+expansions.
        qvec = client.embed(q)
        rankings.append(
            vector_top_k(store.embeddings, qvec, k=per_query_k, mask=effective_mask)
        )
        # BM25 over base+expansions.
        if store.bm25:
            rankings.append(
                store.bm25.top_k(tokenize(q), k=per_query_k, mask=effective_mask)
            )

    # Extra pre-computed vectors (HyDE).
    for vec in (extra_vectors or []):
        rankings.append(
            vector_top_k(store.embeddings, vec, k=per_query_k, mask=effective_mask)
        )

    # Fuse all ranking lists via RRF
    fused = rrf(rankings, k=60)

    return [Candidate(row=row, score=score) for row, score in fused]
