from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Store:
    embeddings: np.ndarray
    bm25: object | None = None
    ready_rows: set[int] = field(default_factory=set)
    writer_lock: threading.Lock = field(default_factory=threading.Lock)


_store: Store | None = None


def get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialized; call set_store() during app startup")
    return _store


def set_store(store: Store) -> None:
    global _store
    _store = store


def reset_store() -> None:
    global _store
    _store = None
