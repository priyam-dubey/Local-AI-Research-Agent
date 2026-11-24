"""
Legacy single-index endpoints, kept for backward compatibility. They now operate
on the 'default' workspace. New code should use /workspaces/{id}/documents.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from app.rag.web_loader import fetch_url_text
from app.rag.vector_store import get_index
from app.services import document_service, repo

router = APIRouter(tags=["Ingestion (legacy – use /workspaces)"], deprecated=True)
WS = repo.DEFAULT_WORKSPACE_ID


class UrlIngestRequest(BaseModel):
    url: str = Field(..., max_length=2000)


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2_000_000)
    source_name: str = Field("manual_input", max_length=200)


def _summary(doc: dict) -> dict:
    return {"source": doc["filename"], "chunks_added": doc["chunks"], "document_id": doc["id"], "status": doc["status"]}


@router.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)) -> dict:
    doc, path = await document_service.save_upload(WS, file)      # type, size, content checks; safe name
    await asyncio.to_thread(document_service.process_document, doc, path)
    doc = repo.get_document(WS, doc["id"])
    if doc["status"] == "failed":
        from app.core.errors import IngestionError
        raise IngestionError(doc["error"] or "Processing failed.")
    return _summary(doc)


@router.post("/ingest/url")
async def ingest_url_endpoint(request: UrlIngestRequest) -> dict:
    title, text = await asyncio.to_thread(fetch_url_text, request.url)       # SSRF-guarded
    doc = await asyncio.to_thread(document_service.ingest_text_document, WS, title, text, "url")
    return _summary(doc)


@router.post("/ingest/text")
async def ingest_text_endpoint(request: TextIngestRequest) -> dict:
    doc = await asyncio.to_thread(document_service.ingest_text_document, WS, request.source_name + ".txt", request.text)
    return _summary(doc)


@router.get("/index/status")
async def index_status() -> dict:
    idx = get_index(WS)
    return {"workspace": WS, "documents": len(idx.doc_ids()), "chunks": len(idx.chunks), "embedding_model": idx.embed_name}


@router.post("/index/reset")
async def index_reset() -> dict:
    for d in repo.list_documents(WS):
        if d["status"] not in ("queued", "processing"):
            await asyncio.to_thread(document_service.delete_document, WS, d["id"])
    return {"status": "reset", "message": "Default workspace cleared."}
