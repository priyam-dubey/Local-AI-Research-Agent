"""Uniform JSON errors for provider failures."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.security import redact
from app.providers.base import ProviderError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code, "message": redact(exc.message), "provider": exc.provider}},
        )
