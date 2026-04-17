"""
Unit tests for the application configuration and environment variable loading.

These tests verify that the Settings class correctly reads and defaults values
from environment variables.
"""

from __future__ import annotations

from app.config import Settings


def test_settings_reads_env(monkeypatch, tmp_path):
    """
    Verify that the Settings class correctly reads values from environment variables
    and sets appropriate defaults for dependent paths.
    """
    # Simulate environment variables
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_DIM", "1024")

    settings = Settings()

    # Direct environment values
    assert settings.mistral_api_key == "test-key"
    assert settings.data_dir == tmp_path
    assert settings.embedding_dim == 1024
    
    # Inferred paths based on DATA_DIR
    assert settings.db_path == tmp_path / "app.sqlite3"
    assert settings.embeddings_path == tmp_path / "embeddings.npy"
    assert settings.bm25_path == tmp_path / "bm25.json"
