"""The research agent can use a workspace's documents, and now really falls back to the web when they are irrelevant."""

from app.agents import researcher
from app.graph import build_graph
from app.providers.factory import LLMSelection, selection_to_config
from app.services import repo
from app.state import initial_state
from tests.helpers import add_doc

PAPER = open("tests/fixtures/corpus/ml_paper.md", "rb").read()


def test_local_search_returns_only_relevant_passages_with_citable_sources():
    ws = repo.create_workspace("r")["id"]
    add_doc(ws, "ml_paper.md", PAPER)
    rel = researcher._local_search("What dataset and how many training images were used?", ws)
    assert rel and rel[0]["type"] == "local" and rel[0]["source"].startswith("ml_paper.md, Section")
    assert researcher._local_search("recommended daily intake of vitamin C", ws) == []   # irrelevant -> web fallback possible


def test_no_workspace_means_no_local_search():
    assert researcher._local_search("anything", None) == [] and not researcher._has_local_docs(None)


def test_graph_uses_local_docs_when_relevant_and_web_when_not(fake_llm, monkeypatch):
    ws = repo.create_workspace("r")["id"]
    add_doc(ws, "ml_paper.md", PAPER)
    web_calls = []

    def fake_web(task):
        web_calls.append(task)
        return [{"content": "web evidence", "source": "https://example.com/x", "score": 1.0, "type": "web"}], \
               [{"title": "t", "href": "https://example.com/x", "body": "web evidence"}]

    monkeypatch.setattr(researcher, "_web_search", fake_web)
    fake_llm.__class__.generate_orig = fake_llm.__class__.generate

    def planner_and_route(messages, **kw):
        kind = __import__("tests.conftest", fromlist=["classify"]).classify(messages)
        if kind == "planner":
            fake_llm.calls.append(kind)
            return '["What dataset and how many training images were used?", "Recommended daily intake of vitamin C"]'
        if kind == "routing":
            fake_llm.calls.append(kind)
            return "local"
        return fake_llm.__class__.generate_orig(fake_llm, messages, **kw)

    monkeypatch.setattr(fake_llm, "generate", planner_and_route)
    cfg = selection_to_config(LLMSelection(provider="ollama"), ws) | {"recursion_limit": 100}
    out = build_graph().invoke(initial_state("dataset question", "s9"), config=cfg)
    assert out["final_report"]
    assert web_calls == ["Recommended daily intake of vitamin C"]             # only the irrelevant step hit the web
    assert any("ml_paper.md" in c for c in out["citations"])                   # the relevant step cited the upload
