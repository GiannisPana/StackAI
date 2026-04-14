"""
Ingestion endpoints for the StackAI RAG application.

This module provides the API router for uploading and processing PDF documents.
It includes validation for file types, size limits, and coordinates the
ingestion pipeline.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import IngestResponse, IngestResultItem
from app.ingestion.pipeline import ingest_pdf
from app.mistral_client import get_mistral_client

router = APIRouter()

# Max allowed size for a single PDF upload (25MB).
MAX_PDF_BYTES = 25 * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(files: list[UploadFile] = File(...)) -> IngestResponse | JSONResponse:
    """
    Ingest one or more PDF documents into the RAG system.

    Processes each uploaded file, performs validation, extracts text,
    and indexes the content for search.

    Args:
        files: A list of UploadFile objects from the request.

    Returns:
        An IngestResponse containing the status of each processed file.
        Returns a 400 JSONResponse if no valid PDFs were found in the request.
    """
    client = get_mistral_client()
    response = IngestResponse()
    saw_pdf_candidate = False

    for upload in files:
        name = upload.filename or "unnamed.pdf"
        content_type = (upload.content_type or "").lower()
        data = await upload.read()
        start = time.monotonic()

        # Initial check by content type header.
        if content_type and "pdf" not in content_type:
            response.failed.append(
                IngestResultItem(filename=name, status="failed", reason="file is not a pdf")
            )
            continue

        # Secondary check by magic bytes.
        if not data[:5].startswith(b"%PDF-"):
            response.failed.append(
                IngestResultItem(filename=name, status="failed", reason="not a valid PDF")
            )
            continue

        saw_pdf_candidate = True

        # Check file size against the limit.
        if len(data) > MAX_PDF_BYTES:
            response.failed.append(
                IngestResultItem(
                    filename=name,
                    status="failed",
                    reason=f"file exceeds {MAX_PDF_BYTES} byte limit",
                )
            )
            continue

        # Execute the ingestion pipeline.
        result = ingest_pdf(client, filename=name, pdf_bytes=data)
        duration_ms = int((time.monotonic() - start) * 1000)
        
        # Build the result item for this file.
        item = IngestResultItem(
            document_id=result.get("document_id"),
            filename=name,
            sha256=result.get("sha256"),
            num_pages=result.get("num_pages"),
            num_chunks=result.get("num_chunks"),
            ocr_pages=result.get("ocr_pages"),
            status=result["status"],
            reason=result.get("reason"),
            duration_ms=duration_ms,
        )

        # Categorize the result based on the pipeline status.
        if result["status"] == "ready":
            response.ingested.append(item)
        elif result["status"] == "skipped":
            response.skipped.append(item)
        else:
            response.failed.append(item)

    # If nothing was successfully processed and all failed, return a 400 error.
    if not saw_pdf_candidate and not response.ingested and not response.skipped and response.failed:
        return JSONResponse(status_code=400, content=response.model_dump())

    return response
