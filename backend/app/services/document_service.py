"""Upload handling, background processing, deletion and cleanup of documents."""

from __future__ import annotations

import time
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from loguru import logger

from app.config import settings
from app.core.errors import AppError, BadRequestError, ConflictError, IngestionError, PayloadTooLargeError
from app.core.security import safe_filename, unique_upload_path
from app.rag.embeddings import get_embedder
from app.rag.ingestion import chunk_sections
from app.rag.loaders import Section, _blocks, detect_type, load_document, validate_file
from app.rag.vector_store import drop_index, get_index
from app.services import repo


async def save_upload(ws_id: str, upload: UploadFile) -> tuple[dict, Path]:
    """Validate type, stream to a temp file under the size limit, sniff content, register the document."""
    if not upload.filename:
        raise BadRequestError("A file is missing its name.")
    file_type = detect_type(upload.filename)
    dest = unique_upload_path(settings.UPLOAD_DIR, upload.filename)
    limit, written = settings.MAX_UPLOAD_MB * 1024 * 1024, 0
    try:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise PayloadTooLargeError(f"'{safe_filename(upload.filename)}' exceeds the {settings.MAX_UPLOAD_MB} MB limit.")
                await f.write(chunk)
        validate_file(dest, file_type)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return repo.create_document(ws_id, safe_filename(upload.filename), file_type, written), dest


def _index_sections(doc: dict, sections: list[Section]) -> None:
    chunks = chunk_sections(sections, doc["id"], doc["filename"])
    if not chunks:
        raise IngestionError("No usable text was found in this document.")
    emb = get_embedder()
    vectors = emb.embed_documents([c["text"] for c in chunks])
    get_index(doc["workspace_id"]).add(chunks, vectors, emb.name)
    repo.update_document(doc["id"], status="ready", pages=len(sections), chunks=len(chunks), error=None)


def process_document(doc: dict, path: Path) -> None:
    """extract -> chunk -> embed -> FAISS. Runs in a worker thread; always removes the temp file."""
    try:
        repo.update_document(doc["id"], status="processing")
        _index_sections(doc, load_document(path, doc["file_type"]))
        logger.info(f"Document {doc['id']} indexed.")
    except AppError as exc:
        repo.update_document(doc["id"], status="failed", error=exc.message)
    except Exception:                               # never log document content
        logger.exception(f"Unexpected failure processing document {doc['id']}")
        repo.update_document(doc["id"], status="failed", error="Unexpected error while processing this file.")
    finally:
        path.unlink(missing_ok=True)


def ingest_text_document(ws_id: str, name: str, text: str, file_type: str = "txt") -> dict:
    """Plain text / fetched web page -> document (inline; used by legacy endpoints and URL import)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()] or [text.strip()]
    doc = repo.create_document(ws_id, safe_filename(name) if file_type != "url" else name[:120], file_type, len(text.encode()))
    try:
        repo.update_document(doc["id"], status="processing")
        _index_sections(doc, _blocks(paras))
    except AppError as exc:
        repo.update_document(doc["id"], status="failed", error=exc.message)
        raise
    return repo.get_document(ws_id, doc["id"])


def delete_document(ws_id: str, doc_id: str) -> None:
    doc = repo.get_document(ws_id, doc_id)
    if doc["status"] in ("queued", "processing"):
        raise ConflictError("This document is still being processed. Try again in a moment.")
    get_index(ws_id).remove_document(doc_id)
    repo.delete_document_row(doc_id)


def delete_workspace(ws_id: str) -> None:
    if ws_id == repo.DEFAULT_WORKSPACE_ID:
        raise BadRequestError("The default workspace cannot be deleted; remove its documents instead.")
    repo.get_workspace(ws_id)
    drop_index(ws_id)
    repo.delete_workspace(ws_id)


def cleanup_stale_uploads(max_age_s: int = 3600) -> int:
    n = 0
    for p in settings.UPLOAD_DIR.glob("*"):
        if p.is_file() and time.time() - p.stat().st_mtime > max_age_s:
            p.unlink(missing_ok=True)
            n += 1
    return n
