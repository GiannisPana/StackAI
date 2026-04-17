"""
Pytest configuration and global fixtures.

This module contains shared fixtures and helper functions for the StackAI
test suite. It handles environment setup, mock client instantiation,
and test client lifecycle management.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

import app.config
from app.deps import reset_store
from app.main import create_app
from tests.fakes.mistral import FakeMistralClient


def make_pdf_bytes(pages: list[str]) -> bytes:
    """
    Generate a PDF in memory for testing.

    Args:
        pages: A list of strings, where each string is the text content
               for a new page in the PDF.

    Returns:
        The raw bytes of the generated PDF document.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            # Insert text at a fixed position
            page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def write_fixture_pdfs(fixtures_dir: Path) -> None:
    """
    Write common PDF test files to a directory.

    Creates 'sample.pdf' with text content and 'scanned.pdf' which is blank
    (simulating a scan that requires OCR).
    """
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Standard PDF with extractable text
    sample_pdf = make_pdf_bytes(
        [
            "Parental leave policy overview.",
            "Employees receive sixteen weeks of paid parental leave.",
            "Eligibility begins after six months of continuous service.",
            "Remote work policy allows hybrid schedules.",
            "Benefits handbook closing page.",
        ]
    )
    (fixtures_dir / "sample.pdf").write_bytes(sample_pdf)

    # Blank PDF to simulate a scanned document
    scanned_pdf = make_pdf_bytes([""])
    (fixtures_dir / "scanned.pdf").write_bytes(scanned_pdf)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Setup a clean test environment.

    Sets required environment variables and resets the global configuration.
    Uses a temporary directory for data storage to ensure isolation between tests.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEBUG", "1")
    
    # Reset settings singleton to pick up new env vars
    app.config._settings = None
    return tmp_path


@pytest.fixture
def fake_mistral() -> FakeMistralClient:
    """
    Provide a FakeMistralClient instance.

    Used for testing components that interact with the Mistral API,
    ensuring deterministic behavior and avoiding network calls.
    """
    return FakeMistralClient(dim=8)


@pytest.fixture
def client(env: Path) -> Iterator[TestClient]:
    """
    Provide a FastAPI TestClient for integration tests.

    Automatically handles application startup and shutdown, and resets
    the storage state (databases/indexes) after each test.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup after test
    reset_store()
    app.config._settings = None
