"""
Central provider selection. This is the ONLY place that decides which concrete
LLM backs a request.

Rules
-----
* The caller's explicit choice always wins; if the chosen provider fails the
  error is surfaced – we never silently fall back to the other provider.
* `settings.DEFAULT_PROVIDER` is used only when the request names no provider.
* API keys are fetched from the SecretStore at call time, never from config
  that flows through LangGraph state.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.config import settings
from app.core.security import secrets
from app.providers.base import BaseLLM, MissingAPIKeyError, UnknownProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider

ProviderName = Literal["ollama", "gemini"]
SUPPORTED_PROVIDERS = ("ollama", "gemini")


class LLMSelection(BaseModel):
    provider: ProviderName
    model: Optional[str] = None


_cache: dict[tuple, BaseLLM] = {}


def clear_provider_cache() -> None:
    """Call whenever the Gemini key changes so no stale client keeps the old key."""
    _cache.clear()


def resolve_selection(provider: Optional[str] = None, model: Optional[str] = None) -> LLMSelection:
    name = provider or settings.DEFAULT_PROVIDER
    if name not in SUPPORTED_PROVIDERS:
        raise UnknownProviderError(
            f"Unknown provider '{name}'. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return LLMSelection(provider=name, model=model or None)


def create_provider(sel: LLMSelection) -> BaseLLM:
    if sel.provider == "ollama":
        return OllamaProvider(model=sel.model)
    if sel.provider == "gemini":
        key = secrets.get_gemini_key()
        if not key:
            raise MissingAPIKeyError(
                "Gemini is selected but no API key is configured. Add one in Settings "
                "or set GEMINI_API_KEY in the backend .env file.", provider="gemini",
            )
        cache_key = ("gemini", sel.model, key)
        if cache_key not in _cache:
            _cache[cache_key] = GeminiProvider(api_key=key, model=sel.model)
        return _cache[cache_key]
    raise UnknownProviderError(f"Unknown provider '{sel.provider}'.")


# ---- LangGraph integration ------------------------------------------------ #
def selection_to_config(sel: LLMSelection) -> dict[str, Any]:
    """RunnableConfig fragment carrying only provider+model (never a key)."""
    return {"configurable": {"llm_provider": sel.provider, "llm_model": sel.model}}


def get_llm(config: Optional[dict] = None) -> BaseLLM:
    """Build the provider for the current graph run from its RunnableConfig."""
    cfg = (config or {}).get("configurable", {}) or {}
    return create_provider(resolve_selection(cfg.get("llm_provider"), cfg.get("llm_model")))


from app.providers import base as _b  # noqa: E402

_ERROR_BY_CODE = {
    c.code: c for c in (
        _b.ProviderUnavailableError, _b.ProviderTimeoutError, _b.MissingAPIKeyError,
        _b.InvalidAPIKeyError, _b.RateLimitError, _b.ModelNotFoundError, _b.UnknownProviderError,
    )
}
