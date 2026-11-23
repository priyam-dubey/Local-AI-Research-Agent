"""Aggregate status for the UI sidebar."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.config import settings
from app.core.security import secrets
from app.providers.ollama_provider import OllamaProvider

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/status")
async def status() -> dict:
    ollama = await asyncio.to_thread(OllamaProvider().health)
    return {
        "ollama": {"available": ollama.available, "detail": ollama.detail, "model": ollama.model, "models": ollama.models},
        "gemini": {"key_configured": secrets.has_gemini_key(), "model": settings.GEMINI_MODEL},
        "vector_store": {"engine": "faiss", "ready": True},
        "embeddings": {"backend": settings.EMBEDDING_BACKEND,
                       "model": settings.EMBED_MODEL if settings.EMBEDDING_BACKEND == "hf" else "hashing-384 (testing only)"},
        "default_provider": settings.DEFAULT_PROVIDER,
    }
