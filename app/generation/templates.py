from __future__ import annotations

DISCLAIMERS = {
    "legal": (
        "Note: this is general information, not legal advice. Consult a qualified attorney "
        "for advice specific to your situation."
    ),
    "medical": (
        "Note: this is general information, not medical advice. Consult a qualified healthcare "
        "professional."
    ),
}

SYSTEM_PROSE = (
    "You are a careful assistant answering questions strictly from the provided numbered "
    "context chunks. Cite every sentence that uses information from a chunk by appending "
    "the chunk number in square brackets, e.g. [1] or [1][2]. If the chunks do not contain "
    "enough information to answer, say so briefly instead of guessing."
)


def _format_chunks(chunks: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"[{index}] {text}" for index, text in chunks)


def build_prose_prompt(
    *,
    query: str,
    chunks: list[tuple[int, str]],
    disclaimer: str | None,
) -> list[dict]:
    parts = [f"Context:\n{_format_chunks(chunks)}", f"Question: {query}"]
    if disclaimer and disclaimer in DISCLAIMERS:
        parts.append(f"Prepend this disclaimer to your answer: {DISCLAIMERS[disclaimer]}")
    return [
        {"role": "system", "content": SYSTEM_PROSE},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
