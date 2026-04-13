from __future__ import annotations

import numpy as np

from app.mistral_client import MistralProtocol


def embed_texts(client: MistralProtocol, texts: list[str]) -> np.ndarray:
    if not texts:
        dim = getattr(client, "dim", None) or getattr(client, "embedding_dim", 0)
        return np.zeros((0, int(dim)), dtype=np.float32)
    return client.embed_batch(texts)
