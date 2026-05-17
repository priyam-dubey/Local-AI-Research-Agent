# AI-Research-Agent/app.py
"""
FastAPI Application Gateway
-----------------------------
Provides the following REST + SSE endpoints:

POST /ingest/file   – Upload and ingest a document into FAISS
POST /ingest/url    – Ingest a web URL into FAISS
POST /ingest/text   – Ingest raw text into FAISS
GET  /index/status  – Current FAISS index summary
POST /index/reset   – Clear the FAISS index

POST /research      – Start a research session (returns session_id)
GET  /research/{id}/stream  – SSE stream of agent messages
GET  /research/{id}/result  – Final report for a completed session

GET  /health        – Health check
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import settings
from graph import get_graph
from rag.ingestion import ingest_file, ingest_url, ingest_text
from rag.vector_store import get_index_summary, reset_vector_store, load_persisted_index
from state import AgentState, initial_state


# --------------------------------------------------------------------------- #
# FastAPI app setup                                                            #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Local AI Research Agent API",
    description="Fully offline, stateful multi-agent research system powered by LangGraph + Ollama.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# In-memory session store                                                      #
# --------------------------------------------------------------------------- #
# Maps session_id → {"state": AgentState, "status": str, "events": list[str]}
_sessions: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Startup event                                                                #
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Local AI Research Agent API...")
    loaded = load_persisted_index()
    if loaded:
        logger.info("Persisted FAISS index restored.")
    logger.info(f"API available at http://{settings.API_HOST}:{settings.API_PORT}/docs")


# --------------------------------------------------------------------------- #
# Pydantic models                                                              #
# --------------------------------------------------------------------------- #
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000, description="Research question")
    session_id: Optional[str] = Field(None, description="Custom session ID (auto-generated if absent)")


class UrlIngestRequest(BaseModel):
    url: str = Field(..., description="Web URL to fetch and ingest")


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source_name: str = Field("manual_input", description="Label for this text source")


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


# --------------------------------------------------------------------------- #
# Health check                                                                 #
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "ok", "model": settings.LLM_MODEL, "timestamp": time.time()}


# --------------------------------------------------------------------------- #
# Ingestion endpoints                                                          #
# --------------------------------------------------------------------------- #
@app.post("/ingest/file", tags=["Ingestion"])
async def ingest_file_endpoint(file: UploadFile = File(...)) -> dict:
    """Upload a PDF, TXT, or MD file for ingestion into the local knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md", ".markdown"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Allowed: .pdf, .txt, .md",
        )

    save_path = settings.UPLOAD_DIR / file.filename
    try:
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        logger.info(f"Uploaded file saved: {save_path}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    result = ingest_file(save_path)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.post("/ingest/url", tags=["Ingestion"])
async def ingest_url_endpoint(request: UrlIngestRequest) -> dict:
    """Fetch a web URL and ingest its content into the knowledge base."""
    result = ingest_url(request.url)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text_endpoint(request: TextIngestRequest) -> dict:
    """Ingest raw text directly into the knowledge base."""
    result = ingest_text(request.text, request.source_name)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# --------------------------------------------------------------------------- #
# Index management                                                             #
# --------------------------------------------------------------------------- #
@app.get("/index/status", tags=["Index"])
async def index_status() -> dict:
    """Return a summary of the current FAISS index."""
    return get_index_summary()


@app.post("/index/reset", tags=["Index"])
async def index_reset() -> dict:
    """Clear the FAISS index entirely."""
    reset_vector_store()
    return {"status": "reset", "message": "FAISS index cleared."}


# --------------------------------------------------------------------------- #
# Research session endpoints                                                   #
# --------------------------------------------------------------------------- #
@app.post("/research", tags=["Research"], response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks) -> ResearchResponse:
    """
    Kick off a new research session.
    The graph runs in the background; poll /research/{id}/stream for live events.
    """
    session_id = request.session_id or str(uuid.uuid4())

    if session_id in _sessions:
        raise HTTPException(status_code=409, detail=f"Session {session_id} already exists.")

    state = initial_state(query=request.query, session_id=session_id)
    _sessions[session_id] = {
        "state": state,
        "status": "running",
        "events": [],
        "final_report": None,
    }

    background_tasks.add_task(_run_graph, session_id, state)

    return ResearchResponse(
        session_id=session_id,
        status="running",
        message=f"Research session started. Stream events at /research/{session_id}/stream",
    )


async def _run_graph(session_id: str, initial: AgentState) -> None:
    """Background task: run the LangGraph and store events into session store."""
    try:
        graph = get_graph()
        session = _sessions[session_id]

        async for chunk in graph.astream(initial, stream_mode="values"):
            # Collect new messages from the chunk
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    event = json.dumps({"role": msg.get("role"), "content": msg.get("content")})
                    session["events"].append(event)
            # Update state snapshot
            session["state"] = chunk

        # Graph complete
        final_state: AgentState = session["state"]
        session["final_report"] = final_state.get("final_report", final_state.get("draft_report", ""))
        session["status"] = "complete"
        logger.info(f"[{session_id}] Research complete.")

    except Exception as exc:
        logger.error(f"[{session_id}] Graph execution error: {exc}")
        _sessions[session_id]["status"] = "error"
        _sessions[session_id]["events"].append(
            json.dumps({"role": "system", "content": f"Error: {exc}"})
        )


@app.get("/research/{session_id}/stream", tags=["Research"])
async def stream_research(session_id: str) -> EventSourceResponse:
    """SSE stream of agent messages for a running research session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        session = _sessions[session_id]
        sent_idx = 0

        while True:
            events = session["events"]
            while sent_idx < len(events):
                yield {"data": events[sent_idx]}
                sent_idx += 1

            if session["status"] in ("complete", "error"):
                yield {"data": json.dumps({"role": "system", "content": f"[{session['status'].upper()}]"})}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/research/{session_id}/result", tags=["Research"])
async def get_result(session_id: str) -> dict:
    """Return the final report for a completed session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    session = _sessions[session_id]
    if session["status"] == "running":
        return {"status": "running", "message": "Research still in progress."}

    state: AgentState = session.get("state", {})
    return {
        "session_id": session_id,
        "status": session["status"],
        "query": state.get("query", ""),
        "final_report": session.get("final_report", ""),
        "citations": state.get("citations", []),
        "quality_score": state.get("quality_score", 0.0),
        "iterations": state.get("iteration", 0),
        "plan": state.get("plan", []),
    }


@app.get("/research/{session_id}/status", tags=["Research"])
async def session_status(session_id: str) -> dict:
    """Quick status check for a research session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "messages_count": len(session["events"]),
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
