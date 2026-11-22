"""Gemini provider using Google's official `google-genai` SDK."""

from __future__ import annotations

import time
from typing import Any, Iterator, Optional

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings
from app.core.security import redact
from app.providers.base import (
    BaseLLM, InvalidAPIKeyError, Message, ModelNotFoundError, ProviderError,
    ProviderHealth, ProviderTimeoutError, ProviderUnavailableError, RateLimitError,
)


def map_gemini_error(exc: Exception) -> ProviderError:
    """Translate SDK / network exceptions into our typed, secret-free errors."""
    if isinstance(exc, ProviderError):
        return exc
    # The SDK may raise exceptions from its own HTTP layer (not only `httpx`), so
    # classify transport errors by walking the class hierarchy by name as well.
    names = {c.__name__ for c in type(exc).__mro__}
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or any("Timeout" in n for n in names):
        return ProviderTimeoutError("Gemini did not respond in time. Try again.", provider="gemini")
    if (
        isinstance(exc, (httpx.TransportError, ConnectionError))
        or names & {"ConnectError", "NetworkError", "ConnectionError", "ProxyError", "RemoteProtocolError"}
    ):
        return ProviderUnavailableError(
            "Cannot reach the Gemini API. Check your internet connection.", provider="gemini"
        )
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None) or 0
        msg = redact(str(getattr(exc, "message", "") or exc))
        low = msg.lower()
        if code in (401, 403) or (code == 400 and ("api key" in low or "api_key" in low)):
            return InvalidAPIKeyError("The Gemini API key is invalid or not permitted.", provider="gemini")
        if code == 404:
            return ModelNotFoundError("The requested Gemini model was not found.", provider="gemini")
        if code == 429:
            return RateLimitError(
                "Gemini rate limit or quota reached. Wait a moment and retry, or check your quota.",
                provider="gemini",
            )
        if code in (408, 504):
            return ProviderTimeoutError("Gemini timed out. Try again.", provider="gemini")
        if code >= 500:
            return ProviderUnavailableError("Gemini is temporarily unavailable.", provider="gemini")
        return ProviderError(f"Gemini request failed ({code}): {msg[:200]}", provider="gemini")
    return ProviderError(f"Unexpected Gemini error: {type(exc).__name__}", provider="gemini")


class GeminiProvider(BaseLLM):
    provider_name = "gemini"
    _RETRYABLE = (RateLimitError, ProviderUnavailableError)

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        backoff: float = 1.0,
        client: Any = None,  # injected in tests
    ) -> None:
        self.model = model or settings.GEMINI_MODEL
        self.max_retries = max_retries
        self.backoff = backoff
        t = timeout if timeout is not None else settings.GEMINI_TIMEOUT
        self._client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(t * 1000))
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split(messages: list[Message]) -> tuple[Optional[str], list[types.Content]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages if m.role != "system"
        ]
        return system, contents

    def _config(self, system, temperature, max_tokens, json_mode) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )

    def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        system, contents = self._split(messages)
        cfg = self._config(system, temperature, max_tokens, json_mode)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=cfg
                )
                text = resp.text
                if not text:
                    raise ProviderError(
                        "Gemini returned no text (the response may have been blocked by safety filters).",
                        provider="gemini",
                    )
                return text
            except Exception as exc:
                err = map_gemini_error(exc)
                if isinstance(err, self._RETRYABLE) and attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise err from None

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        system, contents = self._split(messages)
        cfg = self._config(system, temperature, max_tokens, False)
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=contents, config=cfg
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise map_gemini_error(exc) from None

    def health(self) -> ProviderHealth:
        """Validates the key AND the model with one cheap metadata call (no tokens spent)."""
        try:
            self._client.models.get(model=self.model)
            return ProviderHealth("gemini", True, "Ready", self.model, True, [self.model])
        except Exception as exc:
            err = map_gemini_error(exc)
            return ProviderHealth("gemini", False, err.message, self.model, False, [], err.code)
