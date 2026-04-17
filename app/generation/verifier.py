from __future__ import annotations

import re
from typing import Any

from app.api.schemas import Verification
from app.generation.templates import DISCLAIMERS
from app.mistral_client import MistralProtocol

_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc"}
_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def split_answer_sentences(text: str) -> list[str]:
    # Splits on .!? but (a) keeps known abbreviations like "Dr." attached to the
    # previous clause, and (b) requires the next non-space char to look like a
    # sentence start (uppercase/digit/bracket) to avoid false breaks mid-sentence.
    normalized = " ".join(text.strip().split())
    if not normalized:
        return []

    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char in ".!?":
            fragment = normalized[start : index + 1].strip()
            if char == "." and _ends_with_abbreviation(fragment):
                index += 1
                continue

            end = index + 1
            while end < len(normalized) and normalized[end] in '\'"”’)]}':
                end += 1

            next_index = end
            while next_index < len(normalized) and normalized[next_index].isspace():
                next_index += 1

            if next_index >= len(normalized) or _looks_like_sentence_start(normalized[next_index]):
                sentence = normalized[start:end].strip()
                if sentence:
                    sentences.append(sentence)
                start = next_index
                index = next_index
                continue
        index += 1

    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def parse_citation_tags(text: str) -> list[list[int]]:
    citations_per_sentence: list[list[int]] = []
    for sentence in split_answer_sentences(text):
        citations_per_sentence.append(_citations_in_sentence(sentence))
    return citations_per_sentence


def verify_answer(
    client: MistralProtocol,
    answer: str,
    chunk_lookup: dict[int, str],
) -> Verification:
    sentences = split_answer_sentences(answer)
    if not sentences:
        return Verification(all_supported=False, unsupported_sentences=[])

    ignored_prefix_count = _leading_disclaimer_sentence_count(answer)
    unsupported: set[int] = set()
    cited_sentences: list[tuple[int, str, list[int]]] = []

    for index, sentence in enumerate(sentences):
        if index < ignored_prefix_count:
            continue

        citations = _citations_in_sentence(sentence)
        if not citations:
            if _looks_factual(sentence):
                unsupported.add(index)
            continue

        usable = [citation for citation in citations if chunk_lookup.get(citation, "").strip()]
        if not usable:
            unsupported.add(index)
            continue
        cited_sentences.append((index, sentence, usable))

    entailment = _batched_entailment(client, cited_sentences, chunk_lookup)
    supported_sentence_indexes = {
        sentence_index
        for sentence_index, is_supported in entailment.items()
        if is_supported
    }

    for sentence_index, _, _ in cited_sentences:
        if sentence_index not in supported_sentence_indexes:
            unsupported.add(sentence_index)

    return Verification(
        all_supported=not unsupported,
        unsupported_sentences=sorted(unsupported),
    )


def _ends_with_abbreviation(fragment: str) -> bool:
    match = re.search(r"([A-Za-z]+)\.$", fragment)
    if match is None:
        return False
    return match.group(1).lower() in _ABBREVIATIONS


def _looks_like_sentence_start(char: str) -> bool:
    return char.isupper() or char.isdigit() or char in "[("


def _leading_disclaimer_sentence_count(answer: str) -> int:
    normalized = " ".join(answer.strip().split())
    for disclaimer in DISCLAIMERS.values():
        disclaimer_text = " ".join(disclaimer.split())
        if normalized.startswith(disclaimer_text):
            return len(split_answer_sentences(disclaimer_text))
    return 0


def _looks_factual(sentence: str) -> bool:
    cleaned = _strip_citations(sentence).strip()
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return False
    return bool(re.search(r"[A-Za-z]", cleaned))


def _strip_citations(sentence: str) -> str:
    return " ".join(_CITATION_GROUP_RE.sub("", sentence).split())


def _citations_in_sentence(sentence: str) -> list[int]:
    return [
        int(part.strip())
        for match in _CITATION_GROUP_RE.finditer(sentence)
        for part in match.group(1).split(",")
    ]


def _batched_entailment(
    client: MistralProtocol,
    cited_sentences: list[tuple[int, str, list[int]]],
    chunk_lookup: dict[int, str],
) -> dict[int, bool]:
    """Score all (sentence, cited_chunk) pairs in one LLM call and accept a sentence if any pair is supported."""
    if not cited_sentences:
        return {}

    pair_to_sentence: dict[str, int] = {}
    prompt_lines = [
        "Batched entailment check.",
        "Return STRICT JSON mapping each pair id to true or false.",
    ]

    pair_index = 0
    for sentence_index, sentence, citations in cited_sentences:
        cleaned_sentence = _strip_citations(sentence)
        for citation in citations:
            key = str(pair_index)
            pair_to_sentence[key] = sentence_index
            prompt_lines.extend(
                [
                    f"id: {key}",
                    f"sentence: {cleaned_sentence}",
                    f"context: {chunk_lookup[citation]}",
                ]
            )
            pair_index += 1

    messages = [
        {
            "role": "system",
            "content": (
                "You are an entailment checker. For each pair id, decide whether the "
                "context supports the sentence. Return only JSON mapping ids to booleans."
            ),
        },
        {"role": "user", "content": "\n".join(prompt_lines)},
    ]

    try:
        response = client.chat(messages, response_format={"type": "json_object"})
    except Exception:
        response = None

    parsed = _parse_entailment_response(response, pair_to_sentence.keys())
    supported: dict[int, bool] = {sentence_index: False for sentence_index, _, _ in cited_sentences}
    for key, sentence_index in pair_to_sentence.items():
        if parsed.get(key, False):
            supported[sentence_index] = True
    return supported


def _parse_entailment_response(response: Any, keys: Any) -> dict[str, bool]:
    if not isinstance(response, dict):
        return {str(key): False for key in keys}

    parsed: dict[str, bool] = {}
    for key in keys:
        raw = response.get(str(key), False)
        if isinstance(raw, bool):
            parsed[str(key)] = raw
        elif isinstance(raw, str):
            parsed[str(key)] = raw.strip().lower() == "true"
        else:
            parsed[str(key)] = False
    return parsed
