"""Persistent storage for the BM25 keyword index.

This module provides functions to save and load the BM25 index using a
stage-then-publish pattern to prevent index corruption during writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.retrieval.bm25 import BM25Index


def save_bm25(path: Path, index: BM25Index, publish: bool = True) -> Path | None:
    """Saves the BM25 index to disk.

    By default, it stages the index to a temporary file and then atomically
    publishes it by renaming. This ensures that a crash during the write
    operation does not leave a corrupted main index file.

    Args:
        path: The final path where the index should be saved.
        index: The BM25Index instance to persist.
        publish: If True, atomically replaces the final file. If False,
            only stages the file and returns the path to the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stage_bm25(path, index)
    if publish:
        # Atomic rename to publish the staged file
        os.replace(tmp, path)
        return None
    return tmp


def stage_bm25(path: Path, index: BM25Index) -> Path:
    """Serializes the BM25 index to a temporary '.tmp' file.

    This is the first step of the stage-then-publish pattern.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index.to_dict()), encoding="utf-8")
    return tmp


def load_bm25(path: Path) -> BM25Index:
    """Loads the BM25 index from disk.

    If the file does not exist, a new, empty, finalized index is returned.
    """
    if not path.exists():
        index = BM25Index()
        index.finalize()
        return index
    return BM25Index.from_dict(json.loads(path.read_text(encoding="utf-8")))
