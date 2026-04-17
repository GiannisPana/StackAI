"""
Unit tests for the answer generation component of the RAG pipeline.

These tests verify that the generator correctly uses the provided context to
formulate answers and that citations are correctly extracted from the generated text.
"""

from __future__ import annotations

from app.generation.generator import generate_answer, generate_shaped_answer
from app.generation.verifier import parse_citation_tags as extract_citations
from tests.fakes.mistral import FakeMistralClient


def test_generate_returns_mock_text():
    """
    Verify that generate_answer correctly interacts with the LLM client
    and produces text that includes expected citations.
    """
    fake = FakeMistralClient(dim=4)
    # Register a specific response when "Context:" is seen in the prompt
    fake.register_chat(
        r"Context:",
        "Employees get 16 weeks [1]. Eligibility after 6 months [2].",
    )

    text = generate_answer(fake, query="leave?", chunks=[(1, "a"), (2, "b")], disclaimer=None)

    # The generated text should contain the bracketed citations
    assert "[1]" in text and "[2]" in text


def test_extract_citations_parses_bracket_indices():
    """
    Verify that extract_citations correctly identifies and parses bracketed citation indices
    from a multi-sentence string.
    """
    result = extract_citations("First [1]. Second [2][3]. Third without cite.")
    # Should return a list of lists of integers representing citations per sentence
    assert result == [[1], [2, 3], []]


def test_extract_citations_single_sentence():
    """
    Verify citation extraction from a single sentence containing a citation.
    """
    result = extract_citations("Only sentence [5].")
    assert result == [[5]]


def test_generate_shaped_list_answer_returns_answer_and_structured():
    fake = FakeMistralClient(dim=4)
    fake.register_chat(
        r"items",
        {
            "answer": "- Sixteen weeks [1]",
            "items": [{"text": "Sixteen weeks", "citations": [1]}],
        },
    )

    answer, structured = generate_shaped_answer(
        fake,
        query="list the leave benefits",
        chunks=[(1, "Employees receive sixteen weeks.")],
        format="list",
        disclaimer=None,
    )

    assert answer == "- Sixteen weeks [1]"
    assert structured == {"items": [{"text": "Sixteen weeks", "citations": [1]}]}


def test_generate_shaped_json_answer_preserves_structured_payload():
    fake = FakeMistralClient(dim=4)
    fake.register_chat(
        r"structured",
        {
            "answer": "{\"leave_weeks\": 16}",
            "structured": {"leave_weeks": 16},
        },
    )

    answer, structured = generate_shaped_answer(
        fake,
        query="return the leave policy as json",
        chunks=[(1, "Employees receive sixteen weeks.")],
        format="json",
        disclaimer=None,
    )

    assert answer == "{\"leave_weeks\": 16}"
    assert structured == {"leave_weeks": 16}
