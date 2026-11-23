"""Research sessions (LangGraph multi-agent) + SSE streaming + PDF export."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.providers.factory import resolve_selection
from app.services import repo, research_service as svc
from app.services.export_service import generate_pdf
from app.state import AgentState, initial_state

router = APIRouter(prefix="/research", tags=["Research"])


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000, description="Research question")
    session_id: Optional[str] = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    # Explicit user choice. If omitted, the configured DEFAULT_PROVIDER applies;
    # a failing provider is never replaced by the other one.
    provider: Optional[Literal["ollama", "gemini"]] = None
    model: Optional[str] = Field(None, max_length=100)
    # Optional: use this workspace's uploaded documents as local evidence.
    workspace_id: Optional[str] = Field(None, pattern=r"^[a-z0-9\-]{1,40}$")


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    provider: str
    message: str


def _get(session_id: str) -> dict:
    if session_id not in svc.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return svc.sessions[session_id]


@router.post("", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks) -> ResearchResponse:
    sel = resolve_selection(request.provider, request.model)
    session_id = request.session_id or str(uuid.uuid4())
    if session_id in svc.sessions:
        raise HTTPException(status_code=409, detail=f"Session {session_id} already exists.")

    if request.workspace_id:
        repo.get_workspace(request.workspace_id)      # 404 if unknown
    await svc.preflight(sel)     # meaningful 4xx/5xx before any work starts

    state = initial_state(query=request.query, session_id=session_id)
    svc.new_session(session_id, state, sel, request.workspace_id)
    background_tasks.add_task(svc.run_session, session_id, state, sel, request.workspace_id)
    return ResearchResponse(
        session_id=session_id, status="running", provider=sel.provider,
        message=f"Research session started. Stream events at /research/{session_id}/stream",
    )


@router.get("")
async def research_history(limit: int = 50) -> list[dict]:
    """Completed research sessions (persisted), newest first."""
    return repo.list_research(min(max(limit, 1), 200))


@router.get("/{session_id}/stream")
async def stream_research(session_id: str) -> EventSourceResponse:
    """SSE: `data: {"role","content"}` per agent message; ends with a system [COMPLETE]/[ERROR]."""
    session = _get(session_id)

    async def event_generator() -> AsyncGenerator[dict, None]:
        sent = 0
        while True:
            events = session["events"]
            while sent < len(events):
                yield {"data": events[sent]}
                sent += 1
            if session["status"] in ("complete", "error"):
                yield {"data": json.dumps({"role": "system", "content": f"[{session['status'].upper()}]"})}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/{session_id}/result")
async def get_result(session_id: str) -> dict:
    if session_id not in svc.sessions:                      # after a restart: serve from history
        saved = repo.get_research(session_id)
        if not saved:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        return {"session_id": session_id, "status": saved["status"], "provider": saved["provider"],
                "query": saved["query"], "final_report": saved["report"], "citations": saved["citations"],
                "quality_score": saved["quality"], "iterations": 0, "plan": saved["plan"]}
    session = _get(session_id)
    if session["status"] == "running":
        return {"status": "running", "message": "Research still in progress."}
    state: AgentState = session.get("state", {})
    return {
        "session_id": session_id,
        "status": session["status"],
        "provider": session["provider"],
        "query": state.get("query", ""),
        "final_report": session.get("final_report", ""),
        "citations": state.get("citations", []),
        "quality_score": state.get("quality_score", 0.0),
        "iterations": state.get("iteration", 0),
        "plan": state.get("plan", []),
    }


@router.get("/{session_id}/status")
async def session_status(session_id: str) -> dict:
    s = _get(session_id)
    return {"session_id": session_id, "status": s["status"], "provider": s["provider"],
            "messages_count": len(s["events"])}


@router.get("/{session_id}/report.pdf")
async def export_pdf(session_id: str) -> Response:
    if session_id in svc.sessions:
        s = svc.sessions[session_id]
        report, query, status = s.get("final_report"), s["state"].get("query", ""), s["status"]
    else:
        saved = repo.get_research(session_id)
        if not saved:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        report, query, status = saved["report"], saved["query"], saved["status"]
    if status != "complete" or not report:
        raise HTTPException(status_code=409, detail="No completed report to export yet.")
    pdf = await asyncio.to_thread(generate_pdf, report, query)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="report-{session_id[:8]}.pdf"'})
