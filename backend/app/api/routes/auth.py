"""Sign-in for password-protected deployments."""

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, SecretStr

from app.core import access

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    password: SecretStr = Field(..., min_length=1, max_length=256)


@router.get("/status")
async def status(request: Request) -> dict:
    return {"required": access.required(), "authenticated": access.is_authenticated(request)}


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response) -> dict:
    if access.required():
        access.check_password(body.password.get_secret_value())
        access.set_cookie(response, request)
    return {"authenticated": True}


@router.post("/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(access.COOKIE, path="/")
    return {"authenticated": False}
