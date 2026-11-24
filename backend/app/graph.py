"""
LangGraph Compilation Module
-----------------------------
Defines the StateGraph topology, registers all agent nodes, wires conditional
edges and compiles the executable graph.

Flow (changed from v1 – see CHANGELOG):

  START → planner → researcher ─┐ (loops until every plan step is done)
                        ▲       │
                        │       ▼
                        │   writer_draft → critic
                        │                    │
                        └─ needs more AND the critic added new steps
                                             │ otherwise
                                             ▼
                                        writer_final → END

v1 wrote a full draft after EVERY research step and let the critic approve
after step 1 of N, silently skipping the rest of the plan. Now the draft is
written once the plan is complete, and the critic's GAPS become extra plan
steps (bounded by MAX_ITERATIONS) rather than being ignored.

The LLM provider is NOT part of the graph: it travels in the run's
RunnableConfig (`configurable.llm_provider` / `llm_model`), so one compiled
graph serves Ollama and Gemini alike.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.agents.critic import critic_node
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.writer import writer_draft_node, writer_final_node
from app.config import settings
from app.state import AgentState


# Thin wrappers keep step-level logging and forward the RunnableConfig.
def _planner(state: AgentState, config: RunnableConfig) -> dict:
    logger.debug("Graph: -> planner")
    return planner_node(state, config)


def _researcher(state: AgentState, config: RunnableConfig) -> dict:
    logger.debug("Graph: -> researcher")
    return researcher_node(state, config)


def _writer_draft(state: AgentState, config: RunnableConfig) -> dict:
    logger.debug("Graph: -> writer (draft)")
    return writer_draft_node(state, config)


def _critic(state: AgentState, config: RunnableConfig) -> dict:
    logger.debug("Graph: -> critic")
    return critic_node(state, config)


def _writer_final(state: AgentState, config: RunnableConfig) -> dict:
    logger.debug("Graph: -> writer (final)")
    return writer_final_node(state, config)


# --------------------------------------------------------------------------- #
# Conditional edges                                                            #
# --------------------------------------------------------------------------- #
def _after_researcher(state: AgentState) -> Literal["researcher", "writer_draft"]:
    """Keep researching until every plan step has been executed."""
    if state.get("current_task_index", 0) < len(state.get("plan", [])):
        return "researcher"
    return "writer_draft"


def _after_critic(state: AgentState) -> Literal["researcher", "writer_final"]:
    """
    Go back to research only if the critic wants more AND there are unexecuted
    plan steps (the critic appends its gaps to the plan). Otherwise finalize.
    """
    idx = state.get("current_task_index", 0)
    plan = state.get("plan", [])
    if (
        state.get("needs_more_research", False)
        and state.get("iteration", 0) < settings.MAX_ITERATIONS
        and idx < len(plan)
    ):
        logger.info(f"Graph: critic requested more research ({len(plan) - idx} new step(s)).")
        return "researcher"
    logger.info(f"Graph: finalizing (score={state.get('quality_score', 0):.2f}).")
    return "writer_final"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("researcher", _researcher)
    graph.add_node("writer_draft", _writer_draft)
    graph.add_node("critic", _critic)
    graph.add_node("writer_final", _writer_final)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_conditional_edges(
        "researcher", _after_researcher,
        {"researcher": "researcher", "writer_draft": "writer_draft"},
    )
    graph.add_edge("writer_draft", "critic")
    graph.add_conditional_edges(
        "critic", _after_critic,
        {"researcher": "researcher", "writer_final": "writer_final"},
    )
    graph.add_edge("writer_final", END)

    compiled = graph.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return compiled


_compiled_graph = None


def get_graph():
    """Return the compiled graph singleton (provider-agnostic)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
