"""
Security helpers: server-side secret storage, log redaction, safe filenames.

Design notes
------------
* The Gemini API key lives only in this process's memory (or the GEMINI_API_KEY
  environment variable). It is never put into AgentState, LangGraph config,
  API responses, or logs.
* This is a single-user, local-deployment model. A server-global key is NOT
  safe for multi-user deployment; that needs real auth and per-user secrets.
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from app.config import settings


class SecretStore:
    """Thread-safe in-memory holder for the runtime Gemini API key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gemini_key: str | None = None

    def set_gemini_key(self, key: str) -> None:
        with self._lock:
            self._gemini_key = key.strip()

    def clear_gemini_key(self) -> None:
        with self._lock:
            self._gemini_key = None

    def get_gemini_key(self) -> str | None:
        """Runtime key first, then the environment/.env value."""
        with self._lock:
            if self._gemini_key:
                return self._gemini_key
        env_key = settings.GEMINI_API_KEY
        return env_key.get_secret_value() if env_key else None

    def has_gemini_key(self) -> bool:
        return bool(self.get_gemini_key())


secrets = SecretStore()


def redact(text: str) -> str:
    """Remove any known secret (and key-shaped strings) from text before logging or returning it."""
    key = secrets.get_gemini_key()
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED]", text)


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ \-]")


def safe_filename(name: str, max_len: int = 120) -> str:
    """
    Return a filesystem-safe version of a user-supplied filename.
    Strips directory components (blocks '../' traversal) and unusual characters.
    """
    base = re.split(r"[\\/]", name or "")[-1].strip().strip(".")
    base = _SAFE_CHARS.sub("_", base)
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = (stem or "file")[: max_len - len(ext) - 1]
    return f"{stem}.{ext}" if ext else stem


def unique_upload_path(directory: Path, filename: str) -> Path:
    """Safe, collision-free destination inside `directory` (never overwrites)."""
    path = (directory / f"{uuid.uuid4().hex[:8]}_{safe_filename(filename)}").resolve()
    if directory.resolve() not in path.parents:  # defence in depth
        raise ValueError("Unsafe upload path")
    return path
