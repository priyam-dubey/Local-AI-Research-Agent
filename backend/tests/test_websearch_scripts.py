import sys
import types

import pytest

from app.agents import researcher, websearch
from app.core.errors import EmbeddingError
from app.rag import embeddings
from app.services import repo
from scripts.migrate_v1 import migrate
from tests.helpers import make_pdf


def fake_module(name, results=None, boom=None):
    m = types.ModuleType(name)

    class DDGS:
        def text(self, query, max_results=5):
            if boom:
                raise boom
            return results

    m.DDGS = DDGS
    return m


def test_web_search_normalises_results_from_ddgs(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", fake_module("ddgs", [
        {"title": "A", "href": "https://a.example", "body": "alpha"},
        {"title": "B", "link": "https://b.example", "snippet": "beta"},       # alternate key names
        {"title": "", "body": ""}, "junk"]))
    out = websearch.search_web("q", 5)
    assert out == [{"title": "A", "href": "https://a.example", "body": "alpha"},
                   {"title": "B", "href": "https://b.example", "body": "beta"}]


def test_falls_back_to_legacy_package_when_ddgs_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", None)                              # import raises ImportError
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module("duckduckgo_search", [{"title": "L", "href": "h", "body": "b"}]))
    assert websearch.search_web("q")[0]["title"] == "L"


def test_no_search_package_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", None)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)
    with pytest.raises(websearch.WebSearchUnavailable, match="ddgs"):
        websearch.search_web("q")


def test_researcher_web_search_builds_citable_blocks_and_degrades_on_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", fake_module("ddgs", [{"title": "T", "href": "https://x.example", "body": "evidence"}]))
    blocks, results = researcher._web_search("topic")
    assert blocks == [{"content": "T\nevidence", "source": "https://x.example", "score": 1.0, "type": "web"}]
    monkeypatch.setitem(sys.modules, "ddgs", fake_module("ddgs", boom=RuntimeError("rate limited")))
    assert researcher._web_search("topic") == ([], [])                          # no crash, no web evidence


# ---------------- embedding warm-up state ----------------
def test_embedding_state_reports_loaded_and_load_failures(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "hf")
    embeddings.reset_embedder()
    assert embeddings.embedding_state() == {"loaded": False, "error": None}

    def fail():
        raise EmbeddingError("Could not load embedding model 'x'. Check your internet connection.")

    monkeypatch.setattr(embeddings, "get_embedder", fail)
    embeddings.warm_up()
    st = embeddings.embedding_state()
    assert st["loaded"] is False and "internet connection" in st["error"]
    embeddings.reset_embedder()


def test_system_status_exposes_embedding_state():
    from fastapi.testclient import TestClient
    from app.main import app
    emb = TestClient(app).get("/system/status").json()["embeddings"]
    assert emb["loaded"] is True and emb["error"] is None and emb["backend"] == "hashing"


# ---------------- v1 migration ----------------
def test_migrate_v1_reindexes_original_files_and_skips_the_rest(tmp_path):
    old = tmp_path / "old_uploads"
    old.mkdir()
    (old / "notes.txt").write_text("The beta release is scheduled for 14 March and the budget is 250,000 dollars.")
    (old / "paper.pdf").write_bytes(make_pdf(["Handwritten digits dataset page one.", "Results page two."]))
    (old / "index.pkl").write_bytes(b"\x80\x04pickle")                              # v1 index: not convertible
    (old / "broken.pdf").write_bytes(b"not a pdf")
    r = migrate(old)
    assert sorted(r["imported"]) == ["notes.txt", "paper.pdf"] and r["skipped"] == ["index.pkl"]
    assert len(r["failed"]) == 1 and r["failed"][0].startswith("broken.pdf")
    docs = {d["filename"]: d for d in repo.list_documents(repo.DEFAULT_WORKSPACE_ID)}
    assert docs["paper.pdf"]["status"] == "ready" and docs["paper.pdf"]["pages"] == 2
    assert (old / "notes.txt").exists()                                             # originals are never consumed
