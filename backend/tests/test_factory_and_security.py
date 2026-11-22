import pytest

from app.config import settings
from app.core.security import redact, safe_filename, secrets, unique_upload_path
from app.providers import factory
from app.providers.base import MissingAPIKeyError, UnknownProviderError
from app.providers.factory import LLMSelection, get_llm, resolve_selection, selection_to_config
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider

KEY = "AIzaSyTESTTESTTESTTESTTESTTESTTEST12345"


def test_default_provider_is_local_ollama():
    assert isinstance(get_llm(None), OllamaProvider)


def test_explicit_selection_switches_provider():
    secrets.set_gemini_key(KEY)
    assert isinstance(get_llm(selection_to_config(LLMSelection(provider="gemini"))), GeminiProvider)
    assert isinstance(get_llm(selection_to_config(LLMSelection(provider="ollama", model="llama3"))), OllamaProvider)
    assert get_llm(selection_to_config(LLMSelection(provider="ollama", model="llama3"))).model == "llama3"


def test_gemini_without_key_raises_instead_of_falling_back_to_ollama():
    with pytest.raises(MissingAPIKeyError, match="API key"):
        get_llm(selection_to_config(LLMSelection(provider="gemini")))


def test_default_provider_setting_is_respected_only_when_nothing_selected(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_PROVIDER", "gemini")
    with pytest.raises(MissingAPIKeyError):          # default applies, and it does NOT fall back
        get_llm(None)
    assert isinstance(get_llm(selection_to_config(LLMSelection(provider="ollama"))), OllamaProvider)


def test_unknown_provider_rejected():
    with pytest.raises(UnknownProviderError):
        resolve_selection("openai")


def test_runtime_config_never_contains_the_key():
    secrets.set_gemini_key(KEY)
    cfg = selection_to_config(LLMSelection(provider="gemini"))
    assert KEY not in str(cfg)


def test_env_key_is_used_and_runtime_key_overrides(monkeypatch):
    from pydantic import SecretStr
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("env-key-1234567890"))
    assert secrets.get_gemini_key() == "env-key-1234567890"
    secrets.set_gemini_key(KEY)
    assert secrets.get_gemini_key() == KEY


def test_changing_key_clears_cached_client():
    secrets.set_gemini_key(KEY)
    a = factory.create_provider(LLMSelection(provider="gemini"))
    assert factory.create_provider(LLMSelection(provider="gemini")) is a
    factory.clear_provider_cache()
    assert factory.create_provider(LLMSelection(provider="gemini")) is not a


# ---- security helpers ---------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"),
    ("..\\..\\win.ini", "win.ini"),
    ("my report (final).pdf", "my report _final_.pdf"),
    ("", "file"),
    ("...", "file"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_unique_upload_path_stays_inside_directory(tmp_path):
    p = unique_upload_path(tmp_path, "../../x.txt")
    assert tmp_path.resolve() in p.parents and p.name.endswith("_x.txt")
    assert unique_upload_path(tmp_path, "a.txt") != unique_upload_path(tmp_path, "a.txt")


def test_redact_removes_known_and_key_shaped_secrets():
    secrets.set_gemini_key("custom-secret-value-123")
    out = redact("boom custom-secret-value-123 and AIzaSyABCDEFGHIJKLMNOPQRSTUV")
    assert "custom-secret" not in out and "AIzaSy" not in out
