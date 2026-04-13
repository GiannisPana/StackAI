from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.config import get_settings
from app.deps import Store, reset_store, set_store
from app.storage.db import init_schema
from app.storage.recovery import run_recovery


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_schema()
    embeddings = np.zeros((0, settings.embedding_dim), dtype=np.float32)
    set_store(Store(embeddings=embeddings))
    run_recovery()
    try:
        yield
    finally:
        reset_store()


def create_app() -> FastAPI:
    app = FastAPI(title="StackAI RAG", lifespan=lifespan)
    app.include_router(ingest_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
