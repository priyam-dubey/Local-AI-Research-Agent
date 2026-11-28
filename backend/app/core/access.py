"""
Optional single-password access gate for deployed instances.

* ACCESS_PASSWORD unset  -> everything is open (fine for localhost).
* ACCESS_PASSWORD set    -> API calls need the signed session cookie issued by POST /auth/login.

This protects a single-owner deployment (your Gemini key, your documents). It is NOT multi-user
authentication: everyone who knows the password shares the same workspaces and the same server-side key.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time

from fastapi import Request, Response

from app.config import settings
from app.core.errors import AppError

COOKIE = "ara_session"
SESSION_SECONDS = 7 * 24 * 3600
_MAX_FAILURES, _WINDOW = 10, 300          # 10 wrong passwords per 5 minutes (global) -> lockout until the window passes


class AuthRequiredError(AppError):
    code, http_status = "auth_required", 401


class InvalidPasswordError(AppError):
    code, http_status = "invalid_password", 401


class TooManyAttemptsError(AppError):
    code, http_status = "too_many_attempts", 429


def required() -> bool:
    pw = settings.ACCESS_PASSWORD
    return bool(pw and pw.get_secret_value())


def _password() -> bytes:
    return settings.ACCESS_PASSWORD.get_secret_value().encode()      # type: ignore[union-attr]


def _sign(ts: str) -> str:
    return hmac.new(_password(), f"access-v1:{ts}".encode(), hashlib.sha256).hexdigest()


def make_session_value() -> str:
    ts = str(int(time.time()))
    return f"{ts}.{_sign(ts)}"


def valid_session_value(value: str) -> bool:
    try:
        ts, sig = value.split(".", 1)
        return time.time() - int(ts) < SESSION_SECONDS and hmac.compare_digest(sig, _sign(ts))
    except (ValueError, AttributeError):
        return False


def is_authenticated(request: Request) -> bool:
    return (not required()) or valid_session_value(request.cookies.get(COOKIE, ""))


async def require_auth(request: Request) -> None:
    """FastAPI dependency applied to every router except /auth and /health."""
    if not is_authenticated(request):
        raise AuthRequiredError("Please sign in to continue.")


# ---------------- login throttling ----------------
_lock = threading.Lock()
_failures: list[float] = []


def _recent_failures() -> list[float]:
    cutoff = time.time() - _WINDOW
    _failures[:] = [t for t in _failures if t > cutoff]
    return _failures


def check_password(candidate: str) -> None:
    """Raises on a wrong password or when too many recent failures occurred."""
    with _lock:
        if len(_recent_failures()) >= _MAX_FAILURES:
            raise TooManyAttemptsError("Too many failed sign-in attempts. Wait a few minutes and try again.")
        ok = hmac.compare_digest(candidate.encode(), _password())
        if not ok:
            _failures.append(time.time())
    if not ok:
        raise InvalidPasswordError("Incorrect password.")


def reset_throttle() -> None:
    with _lock:
        _failures.clear()


def set_cookie(response: Response, request: Request) -> None:
    https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
    secure = settings.COOKIE_SECURE == "always" or (settings.COOKIE_SECURE == "auto" and https)
    response.set_cookie(COOKIE, make_session_value(), max_age=SESSION_SECONDS, httponly=True, samesite="lax", secure=secure, path="/")
