"""Ollama provider – talks to the local Ollama REST API over httpx (no cloud calls)."""

from __future__ import annotations

import json
from typing import Iterator, Optional

import httpx

from app.config import settings
from app.providers.base import (
    BaseLLM, Message, ModelNotFoundError, ProviderError, ProviderHealth,
    ProviderTimeoutError, ProviderUnavailableError,
)


def _model_installed(wanted: str, installed: list[str]) -> bool:
    """'mistral' matches 'mistral:latest'; 'llama3:8b' must match exactly."""
    if wanted in installed:
        return True
    if ":" not in wanted:
        return any(name.split(":")[0] == wanted for name in installed)
    return False


class OllamaProvider(BaseLLM):
    provider_name = "ollama"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,  # injected in tests
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT
        self._transport = transport

    # ------------------------------------------------------------------ #
    def _client(self, timeout: Optional[float] = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url, timeout=timeout or self.timeout, transport=self._transport
        )

    def _payload(self, messages, temperature, max_tokens, stream, json_mode=False) -> dict:
        options: dict = {"num_ctx": settings.LLM_CONTEXT_WINDOW}
        options["temperature"] = settings.LLM_TEMPERATURE if temperature is None else temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def _raise_for(self, exc: Exception) -> None:
        if isinstance(exc, httpx.TimeoutException):
            raise ProviderTimeoutError(
                f"Ollama did not respond within {self.timeout:.0f}s. The model may still be "
                "loading, or the prompt is too large for this machine.", provider="ollama"
            ) from exc
        if isinstance(exc, httpx.ConnectError):
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`.",
                provider="ollama",
            ) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{self.model}' is not installed in Ollama. Run `ollama pull {self.model}`.",
                    provider="ollama",
                ) from exc
            raise ProviderUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}.", provider="ollama"
            ) from exc
        raise ProviderError(f"Unexpected Ollama error: {type(exc).__name__}", provider="ollama") from exc

    # ------------------------------------------------------------------ #
    def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        try:
            with self._client() as c:
                r = c.post("/api/chat", json=self._payload(messages, temperature, max_tokens, False, json_mode))
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content", "")
        except (httpx.HTTPError, ValueError) as exc:
            self._raise_for(exc)

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        try:
            with self._client() as c, c.stream(
                "POST", "/api/chat", json=self._payload(messages, temperature, max_tokens, True)
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except (httpx.HTTPError, ValueError) as exc:
            self._raise_for(exc)

    def health(self) -> ProviderHealth:
        try:
            with self._client(timeout=5.0) as c:
                r = c.get("/api/tags")
                r.raise_for_status()
                names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:  # health must never raise
            try:
                self._raise_for(exc)
            except ProviderError as pe:
                return ProviderHealth("ollama", False, pe.message, self.model, False, [], pe.code)
        ready = _model_installed(self.model, names)
        detail = "Ready" if ready else (
            f"Ollama is running but model '{self.model}' is not installed. "
            f"Run `ollama pull {self.model}`."
        )
        return ProviderHealth(
            "ollama", ready, detail, self.model, ready, names, None if ready else "model_not_found"
        )
