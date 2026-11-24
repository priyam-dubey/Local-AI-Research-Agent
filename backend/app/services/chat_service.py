"""Scope resolution + conversation persistence around the grounded-QA pipeline."""

from __future__ import annotations

from typing import Callable, Literal, Optional

from app.core.errors import BadRequestError, ConflictError
from app.providers.base import BaseLLM
from app.rag import document_qa
from app.services import repo

Mode = Literal["document", "all", "compare"]
MAX_COMPARE = 5


def resolve_scope(ws_id: str, mode: Mode, document_ids: list[str]) -> dict[str, str]:
    """Return {doc_id: filename} for the documents this question may use. Always workspace-scoped."""
    if mode == "all":
        docs = repo.list_documents(ws_id, status="ready")
        if not docs:
            raise ConflictError("No processed documents yet. Upload a document and wait until it is ready.")
        return {d["id"]: d["filename"] for d in docs}
    ids = list(dict.fromkeys(document_ids))
    if mode == "document" and len(ids) != 1:
        raise BadRequestError("Choose exactly one document.")
    if mode == "compare" and not (2 <= len(ids) <= MAX_COMPARE):
        raise BadRequestError(f"Choose between 2 and {MAX_COMPARE} documents to compare.")
    out = {}
    for i in ids:
        doc = repo.get_document(ws_id, i)               # 404 if it belongs to another workspace
        if doc["status"] != "ready":
            raise ConflictError(f"'{doc['filename']}' is not ready yet (status: {doc['status']}).")
        out[doc["id"]] = doc["filename"]
    return out


def run_chat(ws_id: str, question: str, mode: Mode, names: dict[str, str], llm: BaseLLM,
             conversation_id: Optional[str] = None, on_stage: Optional[Callable[[str], None]] = None) -> dict:
    conv = repo.get_conversation(ws_id, conversation_id) if conversation_id else None
    history = [{"role": m["role"], "content": m["content"]} for m in repo.list_messages(conv["id"], limit=6)] if conv else []

    if mode == "compare":
        result = document_qa.compare_documents(ws_id, question, doc_names=names, llm=llm, on_stage=on_stage)
    else:
        result = document_qa.answer_question(ws_id, question, doc_ids=set(names), llm=llm, history=history,
                                             on_stage=on_stage, mode=mode)

    conv = conv or repo.create_conversation(ws_id, question, mode)          # only persisted after success
    repo.add_message(conv["id"], "user", question, {"scope": {"mode": mode, "document_ids": list(names)}})
    payload = result.to_dict()
    msg = repo.add_message(conv["id"], "assistant", result.answer, payload)
    return {"conversation_id": conv["id"], "message_id": msg["id"],
            "scope": {"mode": mode, "document_ids": list(names)}, **payload}
