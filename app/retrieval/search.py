from __future__ import annotations

from dataclasses import dataclass

from app.deps import get_store
from app.mistral_client import MistralProtocol
from app.retrieval.bm25 import tokenize
from app.retrieval.fusion import rrf
from app.retrieval.vector_index import top_k as vector_top_k


@dataclass(frozen=True)
class Candidate:
    row: int
    score: float


def hybrid_retrieve(
    client: MistralProtocol,
    *,
    query: str,
    mask: set[int] | None,
    per_query_k: int = 50,
) -> list[Candidate]:
    store = get_store()
    effective_mask = store.ready_rows if mask is None else (mask & store.ready_rows)

    query_vec = client.embed(query)
    cosine = vector_top_k(store.embeddings, query_vec, k=per_query_k, mask=effective_mask)
    bm25_hits = (
        store.bm25.top_k(tokenize(query), k=per_query_k, mask=effective_mask)
        if store.bm25
        else []
    )

    fused = rrf([cosine, bm25_hits], k=60)
    return [Candidate(row=row, score=score) for row, score in fused]
