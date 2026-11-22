"""
Legacy single-index ingestion endpoints (kept for backward compatibility).
Superseded by workspace-scoped /workspaces/{id}/documents endpoints (next phase).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.security import unique_upload_path
from app.rag.ingestion import ingest_file, ingest_text, ingest_url
from app.rag.vector_store import get_index_summary, reset_vector_store

router = APIRouter(tags=["Ingestion (legacy)"])
ALLOWED = (".pdf", ".txt", ".md", ".markdown")


class UrlIngestRequest(BaseModel):
    url: str


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source_name: str = Field("manual_input", max_length=200)


@router.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED)}")

    dest = unique_upload_path(settings.UPLOAD_DIR, file.filename)   # no traversal, no overwrite
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):           # stream; never hold whole file
                written += len(chunk)
                if written > limit:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit.")
                await f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    result = await asyncio.to_thread(ingest_file, dest)             # don't block the event loop
    if "error" in result:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=result["error"])
    result["source"] = file.filename
    return result


@router.post("/ingest/url")
async def ingest_url_endpoint(request: UrlIngestRequest) -> dict:
    result = await asyncio.to_thread(ingest_url, request.url)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.post("/ingest/text")
async def ingest_text_endpoint(request: TextIngestRequest) -> dict:
    result = await asyncio.to_thread(ingest_text, request.text, request.source_name)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/index/status")
async def index_status() -> dict:
    return get_index_summary()


@router.post("/index/reset")
async def index_reset() -> dict:
    reset_vector_store()
    return {"status": "reset", "message": "FAISS index cleared."}
