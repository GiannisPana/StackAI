"""
Query-transform: intent classification and retrieval rewriting.

This module calls the LLM once per query to produce three artefacts used
by the `/query` endpoint before any retrieval takes place:

1. **Intent** — one of ``search``, ``no_search``, or ``refuse``.
   - ``no_search``: greetings, chit-chat, acknowledgements.  The pipeline
     short-circuits and returns a canned reply without touching the index.
   - ``refuse``: requests for personalised legal / medical advice.  The
     pipeline returns a disclaimer without retrieval.
   - ``search``: everything else — proceed normally.

2. **Rewritten query** — a de-personalised, retrieval-optimised reformulation
   of the original.  Used in place of the raw query when calling
   ``hybrid_retrieve``.

3. **Expansion queries** — up to three paraphrases that cover different angles
   of the same information need.  Reserved for future multi-query retrieval.

Resilience contract
-------------------
Any exception raised by the LLM client, or any response that does not contain
the expected ``intent`` key, causes a transparent fall-back: intent is treated
as ``search`` and the original raw query is used for retrieval.  This ensures
that a bad LLM response never takes the endpoint down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.mistral_client import MistralProtocol

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Intent = Literal["search", "no_search", "refuse"]

_INTENT_MAP: dict[str, Intent] = {
    "SEARCH": "search",
    "NO_SEARCH": "no_search",
    "REFUSE": "refuse",
}

_MAX_EXPANSIONS = 3

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You classify a user query against a document knowledge base and rewrite "
    "it for retrieval. "
    "Return STRICT JSON with these exact keys:\n"
    "  intent            — one of SEARCH | NO_SEARCH | REFUSE\n"
    "  sub_intent        — one of factual | list | comparison | definition | "
    "procedural | null\n"
    "  rewritten_query   — a retrieval-optimised reformulation (string; empty "
    "for NO_SEARCH / REFUSE)\n"
    "  expansion_queries — array of up to 3 paraphrases covering different "
    "angles of the same need (empty array for NO_SEARCH / REFUSE)\n\n"
    "Classification rules:\n"
    "  NO_SEARCH : greetings, acknowledgements, chit-chat, empty input.\n"
    "  REFUSE    : requests for personalised legal or medical advice.\n"
    "  SEARCH    : everything else.\n\n"
    "Return only the JSON object, no markdown fences."
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryTransform:
    """
    Immutable result of classifying and rewriting a user query.

    Attributes:
        intent: High-level routing decision for the query pipeline.
        sub_intent: Finer-grained intent used for answer-format selection.
            ``None`` when the LLM returns ``null`` or omits the field.
        rewritten_query: A retrieval-optimised version of the original query.
            For ``no_search`` / ``refuse`` intents this is an empty string.
        expansion_queries: Up to three paraphrases for multi-angle retrieval.
            Always an empty list for ``no_search`` / ``refuse`` intents.
    """

    intent: Intent
    sub_intent: str | None
    rewritten_query: str
    expansion_queries: list[str]


def transform_query(client: MistralProtocol, query: str) -> QueryTransform:
    """
    Classify a user query and rewrite it for retrieval.

    Calls the LLM with a structured-output prompt and parses the JSON response.
    Falls back gracefully to ``search`` intent with the original query if the
    call fails or the response is malformed.

    Args:
        client: An object implementing ``MistralProtocol``.
        query: The raw user query string.

    Returns:
        A :class:`QueryTransform` with ``intent``, ``sub_intent``,
        ``rewritten_query``, and ``expansion_queries`` populated.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Please classify and rewrite this query: {query}",
        },
    ]

    resp: Any = None
    try:
        resp = client.chat(
            messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    except Exception:
        # Network error or any other exception → fall back silently.
        pass

    return _parse_response(resp, raw_query=query)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_response(resp: Any, *, raw_query: str) -> QueryTransform:
    """
    Parse the LLM JSON response into a :class:`QueryTransform`.

    Falls back to a safe default when ``resp`` is not a dict or lacks the
    required ``intent`` key.

    Args:
        resp: The value returned by ``MistralProtocol.chat``.
        raw_query: The original user query, used as the fallback rewritten form.

    Returns:
        A validated :class:`QueryTransform`.
    """
    _fallback = QueryTransform(
        intent="search",
        sub_intent=None,
        rewritten_query=raw_query,
        expansion_queries=[],
    )

    if not isinstance(resp, dict) or "intent" not in resp:
        return _fallback

    # --- intent ---
    intent_raw = str(resp.get("intent", "SEARCH")).strip().upper()
    intent: Intent = _INTENT_MAP.get(intent_raw, "search")  # type: ignore[assignment]

    # --- sub_intent ---
    sub_raw = resp.get("sub_intent")
    if not sub_raw or (isinstance(sub_raw, str) and sub_raw.strip().lower() in ("null", "none", "")):
        sub_intent: str | None = None
    else:
        sub_intent = str(sub_raw).strip()

    # --- rewritten_query ---
    # For NO_SEARCH / REFUSE the LLM legitimately returns ""; honour that.
    # For SEARCH fall back to the raw query if the field is missing or None.
    raw_rewritten = resp.get("rewritten_query")
    if intent in ("no_search", "refuse"):
        rewritten = ""
    elif raw_rewritten:
        rewritten = str(raw_rewritten).strip()
    else:
        rewritten = raw_query

    # --- expansion_queries ---
    raw_exps = resp.get("expansion_queries")
    if not isinstance(raw_exps, list):
        raw_exps = []
    expansions = [str(x).strip() for x in raw_exps if x][:_MAX_EXPANSIONS]

    return QueryTransform(
        intent=intent,
        sub_intent=sub_intent,
        rewritten_query=rewritten,
        expansion_queries=expansions,
    )
