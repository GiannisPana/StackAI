"""
Integration tests for concurrent ingestion behavior in the pipeline layer.
"""
from __future__ import annotations

import threading
import time

import fitz
import numpy as np
import pytest

from app.config import get_settings
from app.deps import Store, get_store, reset_store, set_store
from app.ingestion.pipeline import ingest_pdf
from app.storage.db import get_connection, init_schema
from app.storage.repository import ready_row_set
from app.storage.vector_store import load_matrix
from tests.fakes.mistral import FakeMistralClient


def _make_pdf(text: str, *, pages: int = 1) -> bytes:
    """Build a valid PDF with the given text repeated across pages."""
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{text} page {index + 1}", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Initialize a clean on-disk store for direct pipeline concurrency tests."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    import app.config as app_config

    app_config._settings = None
    init_schema()
    settings = get_settings()
    store = Store(embeddings=np.zeros((0, settings.embedding_dim), dtype=np.float32))
    set_store(store)
    yield
    reset_store()
    app_config._settings = None


def test_concurrent_ingest_threads_publish_consistently(env):
    """Two direct pipeline calls from different threads must leave a consistent store."""
    fake = FakeMistralClient(dim=8)
    barrier = threading.Barrier(2)
    original_embed_batch = fake.embed_batch

    def slow_embed_batch(texts: list[str]) -> np.ndarray:
        barrier.wait(timeout=5)
        time.sleep(0.05)
        return original_embed_batch(texts)

    fake.embed_batch = slow_embed_batch  # type: ignore[assignment]

    results: list[dict] = []

    def worker(name: str, payload: bytes) -> None:
        results.append(ingest_pdf(fake, filename=name, pdf_bytes=payload))

    t1 = threading.Thread(target=worker, args=("1.pdf", _make_pdf("alpha")))
    t2 = threading.Thread(target=worker, args=("2.pdf", _make_pdf("beta")))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(result["status"] for result in results) == ["ready", "ready"]

    settings = get_settings()
    matrix = load_matrix(settings.embeddings_path, expected_dim=8)
    conn = get_connection()
    try:
        ready_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'ready'"
        ).fetchone()[0]
        db_rows = ready_row_set(conn)
    finally:
        conn.close()

    assert ready_count == 2
    assert db_rows == get_store().ready_rows
    assert matrix.shape[0] == len(db_rows)
    assert db_rows == set(range(matrix.shape[0]))


def test_failed_publish_is_reconciled_before_next_writer_rebuild(env, monkeypatch):
    """
    A failed ingest must be marked failed before a later writer rebuilds BM25.

    This reproduces the narrow window where a document is committed as ready,
    publish fails, the lock is released, and a second ingest rebuilds indexes
    before the failed document is rolled back.
    """
    fake = FakeMistralClient(dim=8)
    first_pdf = _make_pdf("first document")
    second_pdf = _make_pdf("second document")

    import app.ingestion.pipeline as pipeline_mod

    entered_mark_failed = threading.Event()
    release_mark_failed = threading.Event()
    original_mark_failed = pipeline_mod._mark_failed

    def blocking_mark_failed(doc_id: int, *, conn=None) -> None:  # noqa: ANN001
        entered_mark_failed.set()
        release_mark_failed.wait(timeout=5)
        original_mark_failed(doc_id, conn=conn)

    monkeypatch.setattr(pipeline_mod, "_mark_failed", blocking_mark_failed)

    original_replace = pipeline_mod.os.replace
    failed_once = {"value": False}
    settings = get_settings()

    def flaky_replace(src: str, dst: str) -> None:
        if not failed_once["value"] and str(dst) == str(settings.embeddings_path):
            failed_once["value"] = True
            raise OSError("forced publish failure after ready commit")
        original_replace(src, dst)

    monkeypatch.setattr(pipeline_mod.os, "replace", flaky_replace)

    first_result: dict[str, object] = {}

    def first_worker() -> None:
        first_result["value"] = ingest_pdf(fake, filename="1.pdf", pdf_bytes=first_pdf)

    t1 = threading.Thread(target=first_worker)
    t1.start()
    assert entered_mark_failed.wait(timeout=5), "expected failing ingest to reach _mark_failed"

    second_result = ingest_pdf(fake, filename="2.pdf", pdf_bytes=second_pdf)
    release_mark_failed.set()
    t1.join()

    assert first_result["value"]["status"] == "failed"
    assert second_result["status"] == "ready"

    postings_rows = {
        row_id
        for postings in get_store().bm25._postings.values()
        for row_id in postings
    }

    conn = get_connection()
    try:
        db_ready = ready_row_set(conn)
    finally:
        conn.close()

    assert db_ready == get_store().ready_rows
    assert postings_rows == db_ready


def test_concurrent_duplicate_ingest_skips_second_writer(env):
    """Two threads ingesting the same PDF should produce one ready result and one skip."""
    fake = FakeMistralClient(dim=8)
    barrier = threading.Barrier(2)
    original_embed_batch = fake.embed_batch

    def slow_embed_batch(texts: list[str]) -> np.ndarray:
        barrier.wait(timeout=5)
        time.sleep(0.05)
        return original_embed_batch(texts)

    fake.embed_batch = slow_embed_batch  # type: ignore[assignment]

    pdf = _make_pdf("duplicate content")
    results: list[dict] = []

    def worker(name: str) -> None:
        results.append(ingest_pdf(fake, filename=name, pdf_bytes=pdf))

    t1 = threading.Thread(target=worker, args=("dup-1.pdf",))
    t2 = threading.Thread(target=worker, args=("dup-2.pdf",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(result["status"] for result in results) == ["ready", "skipped"]
