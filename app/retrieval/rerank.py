"""
LLM-based relevance reranker for the RAG retrieval pipeline.

After the initial hybrid retrieval (vector + BM25 + RRF) produces a ranked
list of candidates, this module asks the LLM to score each candidate's
relevance to the user query on a continuous 0–1 scale.  The result is a
re-ordered candidate list where the most relevant chunks appear first.

Design notes
------------
* The LLM receives a batched prompt listing all candidates with numeric IDs;
  this keeps latency to a single round-trip regardless of candidate count.
* Scores are clamped to [0, 1] to guard against LLM over-confidence.
* Any parsing error or network failure falls back silently to the original
  candidate order, so the endpoint never degrades to a hard error.
* Candidates whose IDs are absent from the LLM response default to a score
  of 0.0 (placed at the bottom after sorting).
"""

from __future__ import annotations

from app.mistral_client import MistralProtocol
from app.retrieval.search import Candidate

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a relevance scorer. Given a query and numbered context chunks, "
    "score each chunk's relevance to the query from 0.0 (completely irrelevant) "
    "to 1.0 (perfectly answers the query). "
    'Return STRICT JSON: {"scores": [{"id": "<number>", "score": <float>}, ...]}\n'
    "Include one entry for every chunk ID. Return only the JSON object, no markdown fences."
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def llm_rerank(
    client: MistralProtocol,
    *,
    query: str,
    candidates: list[Candidate],
    chunk_texts: dict[int, str],
) -> list[Candidate]:
    """
    Re-order *candidates* by LLM-assigned relevance scores.

    Args:
        client: An object satisfying :class:`~app.mistral_client.MistralProtocol`.
        query: The retrieval query (typically the rewritten form).
        candidates: Candidates from hybrid retrieval, in their current rank order.
            The index position of each candidate becomes its ``id`` in the prompt.
        chunk_texts: Mapping from ``Candidate.row`` to the chunk's plain text.
            Rows absent from this dict are included in the prompt as empty strings.

    Returns:
        A new list of :class:`~app.retrieval.search.Candidate` objects, each with
        the LLM relevance score substituted for the original fusion score, sorted
        descending by that score.  Falls back to the original list (unchanged) if
        the LLM call fails or the response is malformed.
    """
    if not candidates:
        return []

    # Build the numbered chunk listing for the prompt.
    lines = [
        f"[{i}] {chunk_texts.get(c.row, '')}"
        for i, c in enumerate(candidates)
    ]
    user_content = (
        f"Query: {query}\n\n"
        "Chunks:\n" + "\n\n".join(lines) + "\n\n"
        "Return relevance scores for all chunk IDs."
    )

    try:
        resp = client.chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
    except Exception:
        # Network / client error — preserve original order.
        return list(candidates)

    return _parse_scores(resp, candidates)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_scores(resp: object, candidates: list[Candidate]) -> list[Candidate]:
    """
    Parse the LLM JSON response and produce a re-scored, sorted candidate list.

    Falls back to the original list when ``resp`` is missing the ``scores`` key
    or is not a dict.

    Args:
        resp: The raw value returned by ``MistralProtocol.chat``.
        candidates: The original candidate list (used for fallback and row lookup).

    Returns:
        Re-scored candidates sorted by descending LLM score.
    """
    if not isinstance(resp, dict) or "scores" not in resp:
        return list(candidates)

    # Parse the id→score map; skip any malformed entries silently.
    score_map: dict[int, float] = {}
    for entry in resp.get("scores", []):
        try:
            idx = int(entry["id"])
            score_map[idx] = float(entry["score"])
        except (KeyError, ValueError, TypeError):
            continue

    # Build rescored candidates; absent IDs default to 0.0.
    rescored: list[Candidate] = []
    for i, candidate in enumerate(candidates):
        raw_score = score_map.get(i, 0.0)
        clamped = max(0.0, min(1.0, raw_score))
        rescored.append(Candidate(row=candidate.row, score=clamped))

    rescored.sort(key=lambda c: -c.score)
    return rescored
