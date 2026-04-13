from __future__ import annotations

import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import IngestResponse, IngestResultItem
from app.ingestion.pipeline import ingest_pdf
from app.mistral_client import get_mistral_client

router = APIRouter()

MAX_PDF_BYTES = 25 * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(files: list[UploadFile] = File(...)) -> IngestResponse | JSONResponse:
    client = get_mistral_client()
    response = IngestResponse()
    saw_pdf_candidate = False

    for upload in files:
        name = upload.filename or "unnamed.pdf"
        content_type = (upload.content_type or "").lower()
        data = await upload.read()
        start = time.monotonic()

        if content_type and "pdf" not in content_type:
            response.failed.append(
                IngestResultItem(filename=name, status="failed", reason="file is not a pdf")
            )
            continue

        if not data[:5].startswith(b"%PDF-"):
            response.failed.append(
                IngestResultItem(filename=name, status="failed", reason="not a valid PDF")
            )
            continue

        saw_pdf_candidate = True

        if len(data) > MAX_PDF_BYTES:
            response.failed.append(
                IngestResultItem(
                    filename=name,
                    status="failed",
                    reason=f"file exceeds {MAX_PDF_BYTES} byte limit",
                )
            )
            continue

        result = ingest_pdf(client, filename=name, pdf_bytes=data)
        duration_ms = int((time.monotonic() - start) * 1000)
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

        if result["status"] == "ready":
            response.ingested.append(item)
        elif result["status"] == "skipped":
            response.skipped.append(item)
        else:
            response.failed.append(item)

    if not saw_pdf_candidate and not response.ingested and not response.skipped and response.failed:
        return JSONResponse(status_code=400, content=response.model_dump())

    return response
