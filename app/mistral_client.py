from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class MistralProtocol(Protocol):
    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...

    def chat(self, messages: list[dict], response_format: Any = None) -> Any: ...

    def ocr(self, pdf_bytes: bytes) -> str: ...


class MistralClient:
    """Thin wrapper around the mistralai SDK."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        chat_model: str,
        rerank_model: str,
        ocr_model: str,
        embedding_dim: int,
    ):
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.rerank_model = rerank_model
        self.ocr_model = ocr_model
        self.embedding_dim = embedding_dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self.embedding_model, inputs=texts)
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def chat(self, messages: list[dict], response_format: Any = None) -> Any:
        import json

        kwargs: dict[str, Any] = {"model": self.chat_model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.complete(**kwargs)
        content = response.choices[0].message.content
        if response_format is not None:
            return json.loads(content)
        return content

    def ocr(self, pdf_bytes: bytes) -> str:
        from base64 import b64encode

        encoded = b64encode(pdf_bytes).decode("ascii")
        response = self._client.ocr.process(
            model=self.ocr_model,
            document={
                "type": "document_base64",
                "document_base64": encoded,
                "document_name": "document.pdf",
            },
        )
        return "\n".join(page.markdown or "" for page in response.pages)


def get_mistral_client() -> MistralProtocol:
    from app.config import get_settings

    settings = get_settings()
    return MistralClient(
        api_key=settings.mistral_api_key,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        rerank_model=settings.rerank_model,
        ocr_model=settings.ocr_model,
        embedding_dim=settings.embedding_dim,
    )
