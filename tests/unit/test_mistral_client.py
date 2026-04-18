from __future__ import annotations

from types import SimpleNamespace

from app.mistral_client import MistralClient


def _wrapper_with_complete(complete):
    client = object.__new__(MistralClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(complete=complete))
    client.chat_model = "mistral-small-latest"
    return client


def test_chat_forwards_explicit_temperature():
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
        )

    client = _wrapper_with_complete(complete)

    result = client.chat([{"role": "user", "content": "hi"}], temperature=0.0)

    assert result == "hello"
    assert captured["temperature"] == 0.0


def test_chat_omits_temperature_when_unspecified():
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )

    client = _wrapper_with_complete(complete)

    result = client.chat(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    assert result == {"ok": True}
    assert "temperature" not in captured
