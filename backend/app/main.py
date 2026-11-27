"""
FastAPI application gateway.

Run from the backend/ directory:   uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.errors import register_error_handlers
from app.api.routes import auth, chat, documents, ingest, research, settings as settings_routes, system
from app.config import settings
from app.core.access import require_auth, required as access_required
from app.rag import embeddings
from app.services import document_service, repo


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting AI Research Agent API...")
    repo.init_db()
    if (n := repo.recover_interrupted()):
        logger.warning(f"{n} document(s) were interrupted by the last shutdown and marked failed.")
    document_service.cleanup_stale_uploads()
    threading.Thread(target=embeddings.warm_up, name="embedding-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="AI Research & Document Intelligence API",
    description="Multi-agent research (LangGraph) with a choice of local Ollama or Gemini.",
    version="2.0.0-dev",
    lifespan=lifespan,
    # when password-protected, do not publish the API schema/docs to the world
    docs_url=None if access_required() else "/docs",
    redoc_url=None if access_required() else "/redoc",
    openapi_url=None if access_required() else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
register_error_handlers(app)
# /auth and /health stay open; everything else requires a session when ACCESS_PASSWORD is set.
app.include_router(auth.router)
_guard = [Depends(require_auth)]
for _router in (settings_routes.router, system.router, documents.router, chat.router, research.router, ingest.router):
    app.include_router(_router, dependencies=_guard)


@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "ok", "default_provider": settings.DEFAULT_PROVIDER, "timestamp": time.time()}




# Serve the built React app (frontend/dist) from the same origin when it exists:
#   cd frontend && npm run build   ->   one process, one port. The SPA uses hash routing, so no fallback rules are needed.
_dist = settings.BASE_DIR.parent / "frontend" / "dist"
if _dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT,
                reload=settings.API_RELOAD, log_level=settings.LOG_LEVEL.lower())

