from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestResultItem(BaseModel):
    document_id: int | None = None
    filename: str
    sha256: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None
    ocr_pages: int | None = None
    status: Literal["ready", "skipped", "failed"]
    reason: str | None = None
    duration_ms: int | None = None


class IngestResponse(BaseModel):
    ingested: list[IngestResultItem] = Field(default_factory=list)
    skipped: list[IngestResultItem] = Field(default_factory=list)
    failed: list[IngestResultItem] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    format: Literal["auto", "prose", "list", "table", "json"] = "auto"
    document_ids: list[int] | None = None
    enable_llm_rerank: bool = True


class Citation(BaseModel):
    index: int
    document_id: int
    filename: str
    page: int
    chunk_id: int
    text: str
    score: float


class Verification(BaseModel):
    all_supported: bool = True
    unsupported_sentences: list[int] = Field(default_factory=list)


class PolicyInfo(BaseModel):
    pii_masked: bool = False
    pii_entities: list[str] = Field(default_factory=list)
    disclaimer: Literal["legal", "medical"] | None = None


class DebugInfo(BaseModel):
    masked_query: str | None = None
    rewritten_query: str | None = None
    expansion_queries: list[str] = Field(default_factory=list)
    hyde_document: str | None = None
    used_hyde: bool = False
    num_candidates: int | None = None
    top_score: float | None = None
    threshold: float | None = None
    latency_ms: dict[str, int] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    format: Literal["prose", "list", "table", "json"] = "prose"
    intent: Literal["search", "no_search", "refuse"] = "search"
    sub_intent: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    structured: dict | None = None
    verification: Verification = Field(default_factory=Verification)
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    refusal_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    debug: DebugInfo | None = None
