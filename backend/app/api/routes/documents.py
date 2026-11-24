"""Workspaces, documents (upload / status / delete), suggested questions."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, File, Path, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import BadRequestError
from app.providers import factory
from app.rag.question_generator import generate_questions
from app.services import document_service, repo

router = APIRouter(prefix="/workspaces", tags=["Workspaces & Documents"])
WS = Annotated[str, Path(pattern=r"^[a-z0-9\-]{1,40}$")]


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


@router.get("")
async def list_workspaces() -> list[dict]:
    return repo.list_workspaces()


@router.post("", status_code=201)
async def create_workspace(body: WorkspaceCreate) -> dict:
    return repo.create_workspace(body.name)


@router.delete("/{ws_id}")
async def delete_workspace(ws_id: WS) -> dict:
    await asyncio.to_thread(document_service.delete_workspace, ws_id)
    return {"deleted": ws_id}


@router.get("/{ws_id}/documents")
async def list_documents(ws_id: WS) -> list[dict]:
    repo.get_workspace(ws_id)
    return repo.list_documents(ws_id)


@router.post("/{ws_id}/documents", status_code=202)
async def upload_documents(ws_id: WS, background: BackgroundTasks, files: list[UploadFile] = File(...)) -> dict:
    """Accepts several files. Returns immediately; poll GET .../documents for `status`."""
    repo.get_workspace(ws_id)
    if len(files) > settings.MAX_UPLOADS_PER_REQUEST:
        raise BadRequestError(f"At most {settings.MAX_UPLOADS_PER_REQUEST} files per upload.")
    accepted, rejected = [], []
    for f in files:
        try:
            doc, path = await document_service.save_upload(ws_id, f)
            background.add_task(document_service.process_document, doc, path)
            accepted.append(doc)
        except Exception as exc:                          # one bad file must not reject the whole batch
            from app.core.errors import AppError
            if not isinstance(exc, AppError):
                raise
            rejected.append({"filename": f.filename, "code": exc.code, "error": exc.message})
    return {"documents": accepted, "rejected": rejected}


@router.get("/{ws_id}/documents/{doc_id}")
async def get_document(ws_id: WS, doc_id: str) -> dict:
    return repo.get_document(ws_id, doc_id)


@router.delete("/{ws_id}/documents/{doc_id}")
async def delete_document(ws_id: WS, doc_id: str) -> dict:
    await asyncio.to_thread(document_service.delete_document, ws_id, doc_id)
    return {"deleted": doc_id}


# ---------------- suggested questions ----------------
class QuestionRequest(BaseModel):
    document_id: Optional[str] = None            # None = whole workspace
    count: int = Field(5, ge=1, le=10)
    provider: Optional[Literal["ollama", "gemini"]] = None
    model: Optional[str] = Field(None, max_length=100)


@router.get("/{ws_id}/questions")
async def stored_questions(ws_id: WS, document_id: str = "") -> dict:
    repo.get_workspace(ws_id)
    return {"questions": repo.list_questions(ws_id, document_id)}


@router.post("/{ws_id}/questions")
async def new_questions(ws_id: WS, body: QuestionRequest) -> dict:
    """Generate (more) questions from the document text with the user-selected provider."""
    from app.services.chat_service import resolve_scope
    names = resolve_scope(ws_id, "document", [body.document_id]) if body.document_id else resolve_scope(ws_id, "all", [])
    llm = factory.create_provider(factory.resolve_selection(body.provider, body.model))
    existing = repo.list_questions(ws_id, body.document_id or "")
    fresh = await asyncio.to_thread(generate_questions, ws_id, set(names), body.count, llm, existing)
    repo.add_questions(ws_id, body.document_id or "", fresh)
    return {"questions": fresh, "all": existing + fresh}
