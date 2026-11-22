"""The real compiled LangGraph run end-to-end with a fake provider (no Ollama / Gemini needed)."""

import pytest

from app.config import settings
from app.graph import build_graph
from app.agents.critic import _parse_critic_response
from app.providers.base import MissingAPIKeyError, ProviderTimeoutError
from app.providers.factory import LLMSelection, selection_to_config
from app.state import initial_state

NEEDS_MORE = "SCORE: 0.3\nVERDICT: NEEDS_MORE_RESEARCH\nFEEDBACK: Thin.\nGAPS:\n- Extra gap topic\n"
PASS = "SCORE: 0.9\nVERDICT: PASS\nFEEDBACK: Good.\nGAPS: None"


def run(sel="ollama", **cfg):
    config = selection_to_config(LLMSelection(provider=sel)) | {"recursion_limit": 100} | cfg
    return build_graph().invoke(initial_state("What is X?", "s1"), config=config)


def roles(state):
    return [m["role"] for m in state["messages"]]


def test_all_four_agents_use_the_selected_provider(fake_llm, no_web):
    out = run("gemini")
    assert out["final_report"].startswith("# Report")
    assert {"planner", "synthesis", "critic", "writer"} <= set(fake_llm.calls)
    assert {s.provider for s in fake_llm.selections} == {"gemini"}      # every agent, same provider


def test_same_graph_runs_with_ollama_too(fake_llm, no_web):
    run("ollama")
    assert {s.provider for s in fake_llm.selections} == {"ollama"}


def test_every_plan_step_is_researched_before_the_critic_runs(fake_llm, no_web):
    out = run()
    r = roles(out)
    assert r.count("researcher") == 2
    assert r.index("critic") > max(i for i, x in enumerate(r) if x == "researcher")


def test_critic_gaps_become_new_research_steps(fake_llm, no_web):
    fake_llm.critic_script = [NEEDS_MORE, PASS]
    out = run()
    assert "Extra gap topic" in out["plan"] and len(out["plan"]) == 3
    assert roles(out).count("researcher") == 3 and out["iteration"] == 2
    assert out["final_report"]


def test_final_report_is_always_written_even_if_critic_never_passes(fake_llm, no_web, monkeypatch):
    monkeypatch.setattr(settings, "MAX_ITERATIONS", 2)
    fake_llm.critic_script = [NEEDS_MORE.replace("Extra gap topic", "gap one"),
                              NEEDS_MORE.replace("Extra gap topic", "gap two")]
    out = run()
    assert out["final_report"], "v1 returned only a draft in this situation"


def test_misconfigured_provider_stops_the_run_instead_of_degrading(monkeypatch, no_web):
    from app.providers import factory

    def boom(sel):
        raise MissingAPIKeyError("no key", provider="gemini")

    monkeypatch.setattr(factory, "create_provider", boom)
    with pytest.raises(MissingAPIKeyError):
        run("gemini")


def test_transient_failures_still_degrade_gracefully(fake_llm, no_web):
    fake_llm.fail_on = {"synthesis": ProviderTimeoutError("slow", provider="x")}
    assert run()["final_report"]


def test_critic_parser_extracts_gaps_and_score():
    parsed = _parse_critic_response(NEEDS_MORE)
    assert parsed["gaps"] == ["Extra gap topic"] and parsed["quality_score"] == 0.3
    assert _parse_critic_response(PASS)["gaps"] == []
