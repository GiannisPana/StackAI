"""
Unit tests for the LLM prompt templates used in answer generation.

These tests verify that the templates correctly format the context chunks,
query, and any optional disclaimers into a coherent prompt for the LLM.
"""

from __future__ import annotations

from app.generation.templates import build_prompt


def test_prose_prompt_includes_all_chunks_with_ids():
    """
    Verify that the prose prompt correctly includes all provided context chunks
    along with their respective IDs for citation purposes.
    """
    chunks = [(1, "chunk one text"), (2, "chunk two text")]

    messages, _ = build_prompt(format="prose", query="what?", chunks=chunks, disclaimer=None)
    # Combine all message contents for easier assertion
    content = " ".join(message["content"] for message in messages)

    # Ensure both chunks are present
    assert "chunk one text" in content
    assert "chunk two text" in content
    # Ensure chunk IDs are present in the format [ID]
    assert "[1]" in content and "[2]" in content
    # The prompt should instruct the LLM to cite its sources
    assert "cite" in content.lower()


def test_prose_prompt_includes_disclaimer():
    """
    Verify that an optional disclaimer is correctly included in the generated prompt.
    """
    messages, _ = build_prompt(format="prose", query="x", chunks=[(1, "t")], disclaimer="legal")
    content = " ".join(message["content"] for message in messages)

    # The disclaimer text should be present in the prompt content
    assert "legal" in content.lower()


def test_list_prompt_requests_structured_items():
    messages, response_format = build_prompt(format="list", query="benefits?", chunks=[(1, "x")], disclaimer=None)
    content = " ".join(message["content"] for message in messages)
    assert "json" in content.lower()
    assert "items" in content.lower()
    assert "citations" in content.lower()
    assert response_format == {"type": "json_object"}


def test_table_prompt_requests_structured_rows():
    messages, response_format = build_prompt(format="table", query="compare leave", chunks=[(1, "x")], disclaimer=None)
    content = " ".join(message["content"] for message in messages)
    assert "json" in content.lower()
    assert "rows" in content.lower()
    assert "citations" in content.lower()
    assert response_format == {"type": "json_object"}


def test_json_prompt_requests_structured_object():
    messages, response_format = build_prompt(format="json", query="summarise", chunks=[(1, "x")], disclaimer=None)
    content = " ".join(message["content"] for message in messages)
    assert "json" in content.lower()
    assert "answer" in content.lower()
    assert "structured" in content.lower()
    assert response_format == {"type": "json_object"}


def test_build_prompt_returns_messages_and_format_spec():
    messages, response_format = build_prompt(
        format="table",
        query="compare leave",
        chunks=[(1, "x")],
        disclaimer=None,
    )

    content = " ".join(message["content"] for message in messages)
    assert "rows" in content.lower()
    assert response_format == {"type": "json_object"}
