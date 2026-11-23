"""Shared fixtures. Nothing here needs Ollama, a Gemini key, or the embedding model."""

from __future__ import annotations

import os

os.environ.setdefault("USER_AGENT", "ai-research-agent-tests")

import pytest

from app.config import settings
from app.core.security import secrets
from app.providers import factory
from app.providers.base import BaseLLM, ProviderHealth


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Fresh SQLite DB + workspace dir per test; hashing embedder (no model download needed)."""
    from app.rag import embeddings, vector_store
    from app.services import repo
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(settings, "WORKSPACES_DIR", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "hashing")
    # thresholds calibrated for the hashing embedder with scripts/calibrate_thresholds.py (see tests/test_calibration.py)
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SIMILARITY", 0.12)
    monkeypatch.setattr(settings, "RETRIEVAL_STRONG_SIMILARITY", 0.45)
    monkeypatch.setattr(settings, "EVIDENCE_MIN_TERM_COVERAGE", 0.5)
    embeddings.reset_embedder()
    vector_store.clear_registry()
    repo.init_db()
    yield
    vector_store.clear_registry()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate secrets/provider cache/sessions between tests."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    secrets.clear_gemini_key()
    factory.clear_provider_cache()
    from app.services import research_service
    research_service.sessions.clear()
    yield
    secrets.clear_gemini_key()
    factory.clear_provider_cache()


def classify(messages) -> str:
    """Which agent is calling? Decided from the system prompt."""
    sys = next((m.content for m in messages if m.role == "system"), "")
    for needle, kind in (
        ("research planning assistant", "planner"),
        ("research routing assistant", "routing"),
        ("rigorous research assistant", "synthesis"),
        ("fact-checking", "critic"),
        ("technical writer", "writer"),
    ):
        if needle in sys:
            return kind
    return "other"


class FakeLLM(BaseLLM):
    """Deterministic stand-in for Ollama/Gemini."""

    def __init__(self, provider="fake", critic_script=None, fail_on=None, healthy=True):
        self.provider_name = provider
        self.model = "fake-model"
        self.calls: list[str] = []
        self.critic_script = list(critic_script or [])
        self.fail_on = fail_on or {}          # kind -> exception instance to raise
        self.healthy = healthy

    def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        kind = classify(messages)
        self.calls.append(kind)
        if kind in self.fail_on:
            raise self.fail_on[kind]
        if kind == "planner":
            return '["Find topic A", "Find topic B"]'
        if kind == "routing":
            return "web"
        if kind == "synthesis":
            return "Synthesized answer. SOURCES: web"
        if kind == "critic":
            if self.critic_script:
                return self.critic_script.pop(0)
            return "SCORE: 0.9\nVERDICT: PASS\nFEEDBACK: Good.\nGAPS: None"
        if kind == "writer":
            return "# Report\n\nBody text.\n\n## References\n- [1] example.com"
        return "ok"

    def stream(self, messages, **kw):
        yield self.generate(messages)

    def health(self) -> ProviderHealth:
        if self.healthy:
            return ProviderHealth(self.provider_name, True, "Ready", "fake-model", True, ["fake-model"])
        return ProviderHealth(self.provider_name, False, "Cannot reach Ollama at http://localhost:11434. "
                              "Start it with `ollama serve`.", "fake-model", False, [], "provider_unavailable")


@pytest.fixture
def fake_llm(monkeypatch):
    """Route every provider lookup to one FakeLLM and record the selections requested."""
    llm = FakeLLM()
    selections: list = []

    def _create(sel):
        selections.append(sel)
        llm.provider_name = sel.provider
        return llm

    monkeypatch.setattr(factory, "create_provider", _create)
    llm.selections = selections
    return llm


@pytest.fixture
def no_web(monkeypatch):
    """Replace DuckDuckGo with canned evidence and ensure no local index is used."""
    from app.agents import researcher

    def _fake_web(task):
        block = {"content": f"Evidence for {task}", "source": "https://example.com/" + task.split()[-1],
                 "score": 1.0, "type": "web"}
        return [block], [{"title": task, "href": block["source"], "body": block["content"]}]

    monkeypatch.setattr(researcher, "_web_search", _fake_web)
    monkeypatch.setattr(researcher, "_has_local_docs", lambda ws: False)
