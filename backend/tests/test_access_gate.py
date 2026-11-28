"""Password gate for deployed instances."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import settings
from app.core import access
from app.main import app

PROTECTED = [("GET", "/settings"), ("GET", "/system/status"), ("GET", "/workspaces"), ("GET", "/workspaces/default/documents"),
             ("GET", "/research"), ("GET", "/index/status"), ("POST", "/workspaces/default/chat"),
             ("POST", "/workspaces/default/documents"), ("POST", "/settings/gemini-key"), ("DELETE", "/settings/gemini-key"),
             ("GET", "/research/abc/stream"), ("POST", "/ingest/text")]


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_PASSWORD", SecretStr("s3cret-pass"))
    access.reset_throttle()
    yield TestClient(app)
    access.reset_throttle()


def test_everything_is_open_when_no_password_is_configured():
    c = TestClient(app)
    assert c.get("/auth/status").json() == {"required": False, "authenticated": True}
    assert c.get("/workspaces").status_code == 200


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_endpoints_reject_anonymous_requests(locked, method, path):
    r = locked.request(method, path, json={} if method == "POST" else None)
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth_required"


def test_health_and_auth_status_stay_public(locked):
    assert locked.get("/health").status_code == 200
    assert locked.get("/auth/status").json() == {"required": True, "authenticated": False}


def test_login_sets_an_httponly_session_that_unlocks_the_api(locked):
    r = locked.post("/auth/login", json={"password": "s3cret-pass"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie.replace("samesite", "")   # plain http in tests
    assert "s3cret-pass" not in r.headers["set-cookie"]                                  # the cookie never contains the password
    assert locked.get("/auth/status").json()["authenticated"] is True
    assert locked.get("/workspaces").status_code == 200


def test_cookie_is_marked_secure_behind_an_https_proxy(locked):
    r = locked.post("/auth/login", json={"password": "s3cret-pass"}, headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower().replace("samesite", "")


def test_wrong_password_is_rejected_and_grants_nothing(locked):
    r = locked.post("/auth/login", json={"password": "wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_password"
    assert "set-cookie" not in r.headers and locked.get("/workspaces").status_code == 401


def test_brute_force_is_throttled_even_for_the_right_password(locked):
    for _ in range(10):
        assert locked.post("/auth/login", json={"password": "nope"}).status_code == 401
    r = locked.post("/auth/login", json={"password": "s3cret-pass"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"


def test_forged_tampered_and_expired_cookies_are_rejected(locked, monkeypatch):
    good = access.make_session_value()
    ts, sig = good.split(".")
    for bad in ("", "garbage", f"{ts}.{'0' * 64}", f"{int(ts) + 1}.{sig}", f"{ts}"):
        locked.cookies.set(access.COOKIE, bad)
        assert locked.get("/workspaces").status_code == 401, bad
    locked.cookies.set(access.COOKIE, good)
    assert locked.get("/workspaces").status_code == 200
    monkeypatch.setattr(access, "SESSION_SECONDS", -1)                                   # expired
    assert locked.get("/workspaces").status_code == 401


def test_changing_the_password_invalidates_existing_sessions(locked, monkeypatch):
    assert locked.post("/auth/login", json={"password": "s3cret-pass"}).status_code == 200
    monkeypatch.setattr(settings, "ACCESS_PASSWORD", SecretStr("a-new-password"))
    assert locked.get("/workspaces").status_code == 401


def test_logout_ends_the_session(locked):
    locked.post("/auth/login", json={"password": "s3cret-pass"})
    locked.post("/auth/logout")
    assert locked.get("/workspaces").status_code == 401


def test_document_chat_and_sse_work_after_login(locked, monkeypatch):
    from app.providers import factory
    from tests.helpers import QALLM
    monkeypatch.setattr(factory, "create_provider", lambda sel: QALLM(lambda k, m: "x"))
    locked.post("/auth/login", json={"password": "s3cret-pass"})
    ws = locked.post("/workspaces", json={"name": "A"}).json()["id"]
    locked.post(f"/workspaces/{ws}/documents", files=[("files", ("a.txt", b"The beta release is scheduled for 14 March this year.", "text/plain"))])
    with locked.stream("POST", f"/workspaces/{ws}/chat/stream", json={"question": "Who won the 2018 football world cup?"}) as r:
        events = [json.loads(l[5:]) for l in r.iter_lines() if l.startswith("data:")]
    assert events and events[-1]["refused"] is True
