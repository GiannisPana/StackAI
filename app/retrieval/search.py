"""High-level search orchestration for the RAG pipeline.

This module coordinates multiple retrieval systems (vector search and BM25)
to perform hybrid retrieval, merging results using Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    per_query_k: int = 50,
) -> list[Candidate]:
    """Perform hybrid retrieval by merging vector search and BM25 results.

    Orchestrates the retrieval pipeline:
    1. Resolve active document mask.
    2. Execute vector search via query embedding.
    3. Execute BM25 search via query tokenization.
    4. Fuse rankings using Reciprocal Rank Fusion (RRF).

    Args:
        client: API client for generating query embeddings.
        query: Raw search query string.
        mask: Optional set of row IDs to restrict the search.
        per_query_k: Number of results to fetch from each retriever.

    Returns:
        A list of hybrid candidates, sorted by RRF score descending.
    """
    store = get_store()
    # Apply global document status mask (e.g., exclude documents being deleted)
    effective_mask = store.ready_rows if mask is None else (mask & store.ready_rows)

    # 1. Vector Retrieval: Dense search based on semantic meaning
    query_vec = client.embed(query)
    cosine = vector_top_k(store.embeddings, query_vec, k=per_query_k, mask=effective_mask)

    # 2. Keyword Retrieval: Sparse search based on exact term matching
    bm25_hits = (
        store.bm25.top_k(tokenize(query), k=per_query_k, mask=effective_mask)
        if store.bm25
        else []
    )

    # 3. Rank Fusion: Merge diverse result sets using RRF
    fused = rrf([cosine, bm25_hits], k=60)

    # Return candidates as structured objects for downstream processing
    return [Candidate(row=row, score=score) for row, score in fused]
