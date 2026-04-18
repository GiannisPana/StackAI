"""
Unit tests for the HyDE (Hypothetical Document Embeddings) helper.
"""
from __future__ import annotations

from tests.fakes.mistral import FakeMistralClient


def test_hyde_expand_returns_hypothetical_doc():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"hypothetical", "GDPR is the EU regulation that governs personal data.")

    from app.retrieval.hyde import hyde_expand

    result = hyde_expand(fake, "what is GDPR?")
    assert result == "GDPR is the EU regulation that governs personal data."


def test_hyde_expand_strips_whitespace():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"hypothetical", "  Some answer.  ")

    from app.retrieval.hyde import hyde_expand

    result = hyde_expand(fake, "tell me something")
    assert result == "Some answer."


def test_hyde_expand_returns_empty_string_on_no_match():
    """When chat returns the default text (no match), should still return a string."""
    fake = FakeMistralClient(dim=8)

    from app.retrieval.hyde import hyde_expand

    result = hyde_expand(fake, "some query")
    assert isinstance(result, str)


def test_hyde_expand_pins_temperature_to_zero():
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, messages, response_format=None, temperature=None):  # noqa: ANN001
            self.calls.append(
                {
                    "messages": messages,
                    "response_format": response_format,
                    "temperature": temperature,
                }
            )
            return " hypothetical answer "

    from app.retrieval.hyde import hyde_expand

    client = RecordingClient()

    result = hyde_expand(client, "some query")  # type: ignore[arg-type]

    assert result == "hypothetical answer"
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.0
