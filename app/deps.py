"""Centralized state management for the StackAI RAG application.

This module provides a singleton 'Store' that holds in-memory search structures
like the embedding matrix and the BM25 index. It facilitates thread-safe
access to these structures across the application.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Store:
    """In-memory search structures for the retrieval engine.

    The store acts as a cache for the persistent storage files (embeddings.npy
    and bm25.json). It is updated when new documents are ingested and published.
    """

    # NumPy matrix of document embeddings, where each row is a chunk vector
    embeddings: np.ndarray

    # BM25 index for keyword-based retrieval
    bm25: object | None = None

    # Set of row indices in 'embeddings' that are fully published and searchable.
    # This prevents searching across chunks from documents that are still being
    # ingested or have partially failed.
    ready_rows: set[int] = field(default_factory=set)

    # Global lock to synchronize writers when updating the store
    writer_lock: threading.Lock = field(default_factory=threading.Lock)


_store: Store | None = None


def get_store() -> Store:
    """Retrieves the global Store instance.

    Raises:
        RuntimeError: If the store hasn't been initialized yet.
    """
    if _store is None:
        raise RuntimeError("Store not initialized; call set_store() during app startup")
    return _store


def set_store(store: Store) -> None:
    """Sets the global Store instance."""
    global _store
    _store = store


def reset_store() -> None:
    """Resets the global Store instance to None, typically for testing."""
    global _store
    _store = None
