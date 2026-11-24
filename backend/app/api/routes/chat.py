"""Document chat: grounded answers with citations, follow-ups, multi-document scopes, SSE stage stream."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, AsyncGenerator, Literal, Optional

from fastapi import APIRouter, Path
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from app.core.errors import AppError
from app.core.security import redact
from app.providers import factory
from app.providers.base import ProviderError
from app.services import chat_service, repo

router = APIRouter(prefix="/workspaces", tags=["Document chat"])
WS = Annotated[str, Path(pattern=r"^[a-z0-9\-]{1,40}$")]


class Scope(BaseModel):
    mode: Literal["document", "all", "compare"] = "all"
    document_ids: list[str] = Field(default_factory=list, max_length=10)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, max_length=32)
    scope: Scope = Field(default_factory=Scope)
    # Explicit user choice; never silently swapped.
    provider: Optional[Literal["ollama", "gemini"]] = None
    model: Optional[str] = Field(None, max_length=100)

    @model_validator(mode="after")
    def _strip(self):
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question must not be blank")
        return self


def _prepare(ws_id: str, req: ChatRequest):
    repo.get_workspace(ws_id)
    names = chat_service.resolve_scope(ws_id, req.scope.mode, req.scope.document_ids)
    llm = factory.create_provider(factory.resolve_selection(req.provider, req.model))   # MissingAPIKeyError etc.
    return names, llm


@router.post("/{ws_id}/chat")
async def chat(ws_id: WS, req: ChatRequest) -> dict:
    names, llm = _prepare(ws_id, req)
    return await asyncio.to_thread(chat_service.run_chat, ws_id, req.question, req.scope.mode, names, llm, req.conversation_id)


@router.post("/{ws_id}/chat/stream")
async def chat_stream(ws_id: WS, req: ChatRequest) -> EventSourceResponse:
    """
    SSE with stage events (retrieving -> generating -> verifying) and one `final` event.
    Answer text is shown only AFTER verification, so unverified model output is never displayed.
    """
    names, llm = _prepare(ws_id, req)            # validation errors are ordinary HTTP errors

    async def events() -> AsyncGenerator[dict, None]:
        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def on_stage(stage: str) -> None:
            loop.call_soon_threadsafe(q.put_nowait, ("stage", {"stage": stage}))

        async def worker() -> None:
            try:
                res = await asyncio.to_thread(chat_service.run_chat, ws_id, req.question, req.scope.mode,
                                              names, llm, req.conversation_id, on_stage)
                await q.put(("final", res))
            except (ProviderError, AppError) as exc:
                await q.put(("error", {"code": exc.code, "message": redact(exc.message)}))
            except Exception:
                logger.exception("chat stream failed")
                await q.put(("error", {"code": "internal_error", "message": "Something went wrong while answering."}))
            finally:
                await q.put(("done", None))

        task = asyncio.create_task(worker())
        try:
            while True:
                kind, payload = await q.get()
                if kind == "done":
                    break
                yield {"event": kind, "data": json.dumps(payload)}
        finally:
            await task

    return EventSourceResponse(events())


@router.get("/{ws_id}/conversations")
async def conversations(ws_id: WS) -> list[dict]:
    repo.get_workspace(ws_id)
    return repo.list_conversations(ws_id)


@router.get("/{ws_id}/conversations/{conv_id}")
async def conversation(ws_id: WS, conv_id: str) -> dict:
    conv = repo.get_conversation(ws_id, conv_id)
    return {**conv, "messages": repo.list_messages(conv_id)}


@router.delete("/{ws_id}/conversations/{conv_id}")
async def delete_conversation(ws_id: WS, conv_id: str) -> dict:
    repo.delete_conversation(ws_id, conv_id)
    return {"deleted": conv_id}
