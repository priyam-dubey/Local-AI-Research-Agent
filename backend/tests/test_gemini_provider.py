from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.core.security import secrets
from app.providers.base import (
    InvalidAPIKeyError, Message, ProviderError, ProviderTimeoutError, RateLimitError,
)
from app.providers.gemini_provider import GeminiProvider

MSGS = [Message("system", "rules"), Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")]
SECRET = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE12345"


def api_error(code, message, status="X"):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


class FakeModels:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def generate_content(self, **kw):
        self.calls.append(kw)
        b = self.behaviours.pop(0) if len(self.behaviours) > 1 else self.behaviours[0]
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(text=b)

    def get(self, **kw):
        b = self.behaviours[0]
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(name=kw["model"])


def gp(*behaviours, **kw):
    client = SimpleNamespace(models=FakeModels(behaviours))
    return GeminiProvider(api_key=SECRET, model="gemini-test", client=client, backoff=0, **kw), client.models


def test_generate_maps_roles_and_system_instruction():
    p, models = gp("answer")
    assert p.generate(MSGS, temperature=0.0, max_tokens=99) == "answer"
    call = models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["config"].system_instruction == "rules"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["config"].max_output_tokens == 99 and call["config"].temperature == 0.0


def test_json_mode_sets_mime_type():
    p, models = gp("{}")
    p.generate(MSGS, json_mode=True)
    assert models.calls[0]["config"].response_mime_type == "application/json"


@pytest.mark.parametrize("code,msg", [
    (400, "API key not valid. Please pass a valid API key."),
    (403, "Permission denied"),
    (401, "Unauthenticated"),
])
def test_invalid_key_variants(code, msg):
    p, _ = gp(api_error(code, msg))
    with pytest.raises(InvalidAPIKeyError) as e:
        p.generate(MSGS)
    assert SECRET not in e.value.message


def test_rate_limit_retries_then_succeeds():
    p, models = gp(api_error(429, "quota"), api_error(429, "quota"), "finally")
    assert p.generate(MSGS) == "finally"
    assert len(models.calls) == 3


def test_rate_limit_gives_up_with_clear_error():
    p, models = gp(api_error(429, "quota"), max_retries=1)
    with pytest.raises(RateLimitError, match="rate limit"):
        p.generate(MSGS)
    assert len(models.calls) == 2


def test_timeout_mapped():
    p, _ = gp(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeoutError):
        p.generate(MSGS)


def test_empty_response_is_an_error_not_silent_empty_string():
    p, _ = gp(None)
    with pytest.raises(ProviderError, match="no text"):
        p.generate(MSGS)


def test_secret_is_redacted_from_error_messages():
    secrets.set_gemini_key(SECRET)
    p, _ = gp(api_error(400, f"Bad request for key {SECRET}"))
    with pytest.raises(ProviderError) as e:
        p.generate(MSGS)
    assert SECRET not in e.value.message and "[REDACTED]" in e.value.message


def test_health_ok():
    p, _ = gp("x")
    h = p.health()
    assert h.available and h.model_ready


def test_health_invalid_key_does_not_raise():
    p, _ = gp(api_error(400, "API key not valid. Please pass a valid API key."))
    h = p.health()
    assert not h.available and h.error_code == "invalid_api_key"


def test_health_unknown_model():
    p, _ = gp(api_error(404, "models/gemini-test is not found"))
    assert p.health().error_code == "model_not_found"


def test_foreign_sdk_network_errors_are_recognised_by_class_name():
    """Regression: the SDK's own ConnectError is not httpx.ConnectError (found in smoke test)."""
    class ConnectError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    from app.providers.base import ProviderUnavailableError
    p, _ = gp(ConnectError("boom"), max_retries=0)
    with pytest.raises(ProviderUnavailableError, match="internet"):
        p.generate(MSGS)
    p, _ = gp(ReadTimeout("slow"), max_retries=0)
    with pytest.raises(ProviderTimeoutError):
        p.generate(MSGS)
