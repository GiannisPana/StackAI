"""Configuration management for the StackAI RAG application.

This module defines the global settings used throughout the application,
leveraging Pydantic for environment-based configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings and environment configurations.

    This class manages API keys, model identifiers, retrieval parameters,
    and storage paths. It automatically loads values from a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mistral_api_key: str = Field(default="")
    data_dir: Path = Field(default=Path("./data"))
    debug: bool = False

    # Model identifiers for various Mistral AI services
    embedding_model: str = "mistral-embed"
    embedding_dim: int = 1024
    chat_model: str = "mistral-small-latest"
    rerank_model: str = "mistral-small-latest"
    ocr_model: str = "mistral-ocr-latest"

    # Retrieval and chunking parameters
    threshold_low: float = 0.45
    threshold_high: float = 0.55
    spread_min: float = 0.15
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    mmr_lambda: float = 0.7
    max_tokens_per_chunk: int = 512
    chunk_overlap_tokens: int = 64
    max_pdf_bytes: int = 25 * 1024 * 1024

    @property
    def db_path(self) -> Path:
        """The absolute path to the SQLite database file."""
        return self.data_dir / "app.sqlite3"

    @property
    def embeddings_path(self) -> Path:
        """The absolute path to the NumPy file storing document embeddings."""
        return self.data_dir / "embeddings.npy"

    @property
    def bm25_path(self) -> Path:
        """The absolute path to the JSON file storing the BM25 index."""
        return self.data_dir / "bm25.json"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Returns the singleton Settings instance, initializing it if necessary.

    The first call to this function will create the data directory if it
    does not already exist.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure the base directory for persistent storage exists
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
    return _settings
