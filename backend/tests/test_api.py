import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.security import secrets
from app.main import app
from app.services import research_service as svc
from app.services.export_service import generate_pdf
from app.services.research_service import collect_new_events
from app.providers.base import ProviderHealth

KEY = "AIzaSyAPITESTAPITESTAPITESTAPITEST123"
client = TestClient(app)


# ---- SSE duplicate-event bug -------------------------------------------- #
def test_collect_new_events_does_not_resend_earlier_messages():
    events: list[str] = []
    msgs = [{"role": "planner", "content": "p"}]
    sent = collect_new_events(events, {"messages": list(msgs)}, 0)
    msgs.append({"role": "researcher", "content": "r"})
    sent = collect_new_events(events, {"messages": list(msgs)}, sent)    # cumulative state, as astream(values) yields
    sent = collect_new_events(events, {"messages": list(msgs)}, sent)    # no new message at this step
    assert [json.loads(e)["role"] for e in events] == ["planner", "researcher"]


# ---- provider preflight / explicit selection ----------------------------- #
def test_gemini_without_key_returns_clear_400():
    r = client.post("/research", json={"query": "What is quantum computing?", "provider": "gemini"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "missing_api_key" and "API key" in err["message"]


def test_unreachable_ollama_returns_503_with_hint(fake_llm):
    fake_llm.healthy = False
    r = client.post("/research", json={"query": "What is quantum computing?", "provider": "ollama"})
    assert r.status_code == 503 and "ollama serve" in r.json()["error"]["message"]


def test_invalid_provider_name_rejected_by_validation():
    assert client.post("/research", json={"query": "What is quantum computing?", "provider": "openai"}).status_code == 422


# ---- Gemini key endpoints ------------------------------------------------- #
class _Health:
    def __init__(self, h): self._h = h
    def __call__(self, api_key, **kw): return self
    def health(self): return self._h


def test_invalid_gemini_key_is_rejected_and_not_stored(monkeypatch):
    bad = ProviderHealth("gemini", False, "The Gemini API key is invalid or not permitted.", "m", False, [], "invalid_api_key")
    monkeypatch.setattr("app.api.routes.settings.GeminiProvider", _Health(bad))
    r = client.post("/settings/gemini-key", json={"api_key": KEY})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert KEY not in r.text and not secrets.has_gemini_key()


def test_valid_key_is_stored_server_side_and_never_returned(monkeypatch):
    ok = ProviderHealth("gemini", True, "Ready", "m", True, ["m"])
    monkeypatch.setattr("app.api.routes.settings.GeminiProvider", _Health(ok))
    r = client.post("/settings/gemini-key", json={"api_key": KEY})
    assert r.status_code == 200 and r.json()["valid"] is True and KEY not in r.text
    s = client.get("/settings")
    assert s.json()["providers"]["gemini"]["key_configured"] is True and KEY not in s.text
    assert client.get("/settings/providers/gemini/health").status_code == 200
    client.delete("/settings/gemini-key")
    assert not secrets.has_gemini_key()


def test_short_key_fails_validation():
    assert client.post("/settings/gemini-key", json={"api_key": "x"}).status_code == 422


def test_health_endpoint_reports_missing_key_without_error():
    body = client.get("/settings/providers/gemini/health").json()
    assert body["available"] is False and body["error_code"] == "missing_api_key"


# ---- full research flow through the API (SSE + result + PDF) --------------- #
def test_research_flow_streams_unique_events_and_exports_pdf(fake_llm, no_web):
    r = client.post("/research", json={"query": "What is quantum computing?", "provider": "gemini"})
    assert r.status_code == 200 and r.json()["provider"] == "gemini"
    sid = r.json()["session_id"]

    with client.stream("GET", f"/research/{sid}/stream") as resp:
        data = [json.loads(l[5:]) for l in resp.iter_lines() if l.startswith("data:")]
    contents = [d["content"] for d in data]
    assert "Gemini" in contents[0]                      # user is told which provider is running
    assert contents[-1] == "[COMPLETE]"
    assert len(contents) == len(set(contents)), "duplicate SSE events"
    assert {"planner", "researcher", "critic", "writer"} <= {d["role"] for d in data}

    res = client.get(f"/research/{sid}/result").json()
    assert res["status"] == "complete" and res["final_report"].startswith("# Report") and res["provider"] == "gemini"
    pdf = client.get(f"/research/{sid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_provider_failure_mid_run_is_reported_as_error_event(fake_llm, no_web):
    from app.providers.base import InvalidAPIKeyError
    fake_llm.fail_on = {"planner": InvalidAPIKeyError("The Gemini API key is invalid or not permitted.")}
    sid = client.post("/research", json={"query": "What is quantum computing?"}).json()["session_id"]
    assert svc.sessions[sid]["status"] == "error"
    assert "invalid" in svc.sessions[sid]["events"][-1]


def test_pdf_export_survives_markup_characters():
    md = "# R&D <overview>\n## A & B\n- item <b>\n> quote & more\nplain 5 < 6 & 7 > 3"
    assert generate_pdf(md, "q & <x>").startswith(b"%PDF")


# ---- upload hardening (legacy endpoint, now backed by the default workspace) ---- #
def test_upload_path_traversal_is_neutralised(tmp_path):
    body = b"Hello world. This is a plain text document with enough words to be indexed properly."
    r = client.post("/ingest/file", files={"file": ("../../evil.txt", body, "text/plain")})
    assert r.status_code == 200 and r.json()["source"] == "evil.txt"
    assert not (tmp_path.parent / "evil.txt").exists()
    assert list(settings.UPLOAD_DIR.iterdir()) == []          # temp file cleaned up


def test_upload_size_limit_enforced_and_partial_file_removed(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    r = client.post("/ingest/file", files={"file": ("a.txt", b"x" * 10, "text/plain")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_upload_rejects_disallowed_type():
    r = client.post("/ingest/file", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
