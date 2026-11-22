"""
Provider abstraction shared by every agent and the document-QA pipeline.

Agents only ever see `BaseLLM`; they do not know whether Ollama or Gemini is
behind it. Embeddings are intentionally NOT part of this interface: they are
local (HuggingFace) and independent of the selected generation provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


# --------------------------------------------------------------------------- #
# Errors – each carries a stable machine-readable `code` and a message that     #
# is safe to show to the user (never contains secrets).                         #
# --------------------------------------------------------------------------- #
class ProviderError(Exception):
    code = "provider_error"
    http_status = 502

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    code, http_status = "provider_unavailable", 503


class ProviderTimeoutError(ProviderError):
    code, http_status = "timeout", 504


class MissingAPIKeyError(ProviderError):
    code, http_status = "missing_api_key", 400


class InvalidAPIKeyError(ProviderError):
    code, http_status = "invalid_api_key", 401


class RateLimitError(ProviderError):
    code, http_status = "rate_limited", 429


class ModelNotFoundError(ProviderError):
    code, http_status = "model_not_found", 404


class UnknownProviderError(ProviderError):
    code, http_status = "unknown_provider", 400


@dataclass
class ProviderHealth:
    provider: str
    available: bool
    detail: str
    model: str = ""
    model_ready: bool = False
    models: list[str] = field(default_factory=list)
    error_code: Optional[str] = None


class BaseLLM(ABC):
    provider_name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Return the full completion text. Raises ProviderError subclasses."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield completion text incrementally."""

    @abstractmethod
    def health(self) -> ProviderHealth:
        """Check reachability / credentials / model availability. Never raises."""


# Errors that mean "this provider is misconfigured/unreachable". Agents must NOT
# paper over these with fallback content – the run should stop with a clear message.
FATAL_PROVIDER_ERRORS = (
    MissingAPIKeyError, InvalidAPIKeyError, ModelNotFoundError,
    ProviderUnavailableError, UnknownProviderError,
)
