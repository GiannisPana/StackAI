from __future__ import annotations

import json
import os
from pathlib import Path

from app.retrieval.bm25 import BM25Index


def save_bm25(path: Path, index: BM25Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index.to_dict()), encoding="utf-8")
    os.replace(tmp, path)


def load_bm25(path: Path) -> BM25Index:
    if not path.exists():
        index = BM25Index()
        index.finalize()
        return index
    return BM25Index.from_dict(json.loads(path.read_text(encoding="utf-8")))
