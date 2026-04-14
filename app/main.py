"""
Main entry point for the StackAI RAG FastAPI application.

This module initializes the FastAPI app, configures the API routers,
sets up the database schema, and manages the application lifespan
including resource initialization and cleanup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.documents import router as documents_router
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.config import get_settings
from app.deps import Store, reset_store, set_store
from app.storage.db import init_schema
from app.storage.recovery import run_recovery


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifespan.

    Performs startup tasks such as initializing the database schema,
    loading initial state into the Store, and running recovery processes.
    Ensures proper cleanup on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    settings = get_settings()
    
    # Ensure database tables exist.
    init_schema()
    
    # Initialize the global in-memory store for embeddings.
    embeddings = np.zeros((0, settings.embedding_dim), dtype=np.float32)
    set_store(Store(embeddings=embeddings))
    
    # Run recovery to sync in-memory state with the database.
    run_recovery()
    
    try:
        yield
    finally:
        # Clear resources on shutdown.
        reset_store()


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Returns:
        A configured FastAPI app instance.
    """
    app = FastAPI(title="StackAI RAG", lifespan=lifespan)
    
    # Include API routers.
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(documents_router)

    # Serve the chat UI.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        if static_dir.exists():
            return FileResponse(static_dir / "index.html")
        return {"error": "UI static files not found"}

    @app.get("/health")
    def health():
        """
        Health check endpoint.

        Returns:
            A simple status message indicating the service is running.
        """
        return {"status": "ok"}

    return app


# Create the application instance.
app = create_app()
