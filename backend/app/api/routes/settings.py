"""Provider selection, health checks and Gemini key management."""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, SecretStr

from app.config import settings
from app.core.security import secrets
from app.providers.base import ProviderError
from app.providers import factory
from app.providers.factory import _ERROR_BY_CODE, clear_provider_cache, resolve_selection
from app.providers.gemini_provider import GeminiProvider

router = APIRouter(prefix="/settings", tags=["Settings"])


class GeminiKeyRequest(BaseModel):
    api_key: SecretStr = Field(..., min_length=10, max_length=256)


@router.get("")
async def get_settings() -> dict:
    """Current configuration. Never includes any key material."""
    return {
        "default_provider": settings.DEFAULT_PROVIDER,
        "providers": {
            "ollama": {"model": settings.LLM_MODEL},
            "gemini": {"model": settings.GEMINI_MODEL, "key_configured": secrets.has_gemini_key()},
        },
    }


@router.get("/providers/{provider}/health")
async def provider_health(provider: Literal["ollama", "gemini"], model: Optional[str] = Query(None)) -> dict:
    """Availability check. Always 200; read `available` / `error_code`."""
    try:
        llm = factory.create_provider(resolve_selection(provider, model))
    except ProviderError as exc:   # e.g. no Gemini key yet
        return {"provider": provider, "available": False, "detail": exc.message, "error_code": exc.code,
                "model": model or "", "model_ready": False, "models": []}
    h = await asyncio.to_thread(llm.health)
    return h.__dict__


@router.post("/gemini-key")
async def set_gemini_key(body: GeminiKeyRequest) -> dict:
    """
    Validate the key against Google, then keep it in backend memory only.
    The key is never echoed back, stored in the browser, or written to disk/logs.
    """
    key = body.api_key.get_secret_value().strip()
    health = await asyncio.to_thread(GeminiProvider(api_key=key, max_retries=0).health)
    if health.available or health.error_code == "model_not_found":
        secrets.set_gemini_key(key)
        clear_provider_cache()
        warning = None if health.available else (
            f"Key accepted, but model '{settings.GEMINI_MODEL}' was not found. Set GEMINI_MODEL in .env."
        )
        return {"valid": True, "configured": True, "warning": warning}
    raise _ERROR_BY_CODE.get(health.error_code or "", ProviderError)(health.detail, provider="gemini")


@router.delete("/gemini-key")
async def delete_gemini_key() -> dict:
    secrets.clear_gemini_key()
    clear_provider_cache()
    return {"configured": secrets.has_gemini_key()}   # may stay true if set via environment
