import json

import httpx
import pytest

from app.providers.base import (
    Message, ModelNotFoundError, ProviderTimeoutError, ProviderUnavailableError,
)
from app.providers.ollama_provider import OllamaProvider

MSGS = [Message("system", "be brief"), Message("user", "hi")]


def provider(handler, **kw):
    return OllamaProvider(model="mistral", base_url="http://ollama.test", transport=httpx.MockTransport(handler), **kw)


def test_generate_sends_expected_payload_and_returns_text():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hello"}})

    out = provider(handler).generate(MSGS, temperature=0.1, max_tokens=50, json_mode=True)
    assert out == "hello"
    b = seen["body"]
    assert b["model"] == "mistral" and b["stream"] is False and b["format"] == "json"
    assert b["options"]["temperature"] == 0.1 and b["options"]["num_predict"] == 50
    assert [m["role"] for m in b["messages"]] == ["system", "user"]


def test_connection_refused_gives_actionable_error():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError, match="ollama serve"):
        provider(handler).generate(MSGS)


def test_timeout_is_reported_as_timeout():
    def handler(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        provider(handler, timeout=1).generate(MSGS)


def test_missing_model_tells_user_how_to_pull_it():
    with pytest.raises(ModelNotFoundError, match="ollama pull mistral"):
        provider(lambda req: httpx.Response(404, json={"error": "model not found"})).generate(MSGS)


def test_health_matches_latest_tag():
    p = provider(lambda req: httpx.Response(200, json={"models": [{"name": "mistral:latest"}, {"name": "llama3:8b"}]}))
    h = p.health()
    assert h.available and h.model_ready and "llama3:8b" in h.models


def test_health_reports_missing_model_without_raising():
    p = provider(lambda req: httpx.Response(200, json={"models": [{"name": "llama3:8b"}]}))
    h = p.health()
    assert not h.available and h.error_code == "model_not_found" and "ollama pull" in h.detail


def test_health_never_raises_when_unreachable():
    def handler(req):
        raise httpx.ConnectError("refused")

    h = provider(handler).health()
    assert not h.available and h.error_code == "provider_unavailable"


def test_stream_yields_incremental_chunks():
    lines = "\n".join(json.dumps(c) for c in [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ])
    assert "".join(provider(lambda req: httpx.Response(200, content=lines)).stream(MSGS)) == "Hello"
