# AI-Research-Agent/graph.py
"""
LangGraph Compilation Module
-----------------------------
Defines the StateGraph topology, registers all four agent nodes,
wires conditional edges, and compiles the executable graph.

Flow Diagram:
  START → planner → researcher → writer(draft) → critic
                        ↑                              |
                        |          needs_more=True      |
                        └──────────────────────────────┘
                                       |
                             needs_more=False
                                       ↓
                              writer(final) → END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END
from loguru import logger

from config import settings
from state import AgentState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.critic import critic_node
from agents.writer import writer_node


# --------------------------------------------------------------------------- #
# Node wrappers with step-level logging                                        #
# --------------------------------------------------------------------------- #
def _planner(state: AgentState) -> dict:
    logger.debug("Graph: → planner node")
    return planner_node(state)


def _researcher(state: AgentState) -> dict:
    logger.debug("Graph: → researcher node")
    return researcher_node(state)


def _writer(state: AgentState) -> dict:
    logger.debug("Graph: → writer node")
    return writer_node(state)


def _critic(state: AgentState) -> dict:
    logger.debug("Graph: → critic node")
    return critic_node(state)


# --------------------------------------------------------------------------- #
# Conditional edge logic                                                       #
# --------------------------------------------------------------------------- #
def _should_continue_research(state: AgentState) -> Literal["researcher", "writer_final"]:
    """
    After the Critic evaluates:
      - If quality is sufficient → go to writer_final to produce the final report
      - If more research is needed AND we haven't hit MAX_ITERATIONS → loop back
    MAX_ITERATIONS is enforced inside critic_node; by the time we reach here
    with needs_more_research=False it is always safe to finalize.
    """
    needs_more: bool = state.get("needs_more_research", True)
    iteration: int = state.get("iteration", 0)

    if not needs_more or iteration >= settings.MAX_ITERATIONS:
        logger.info(f"Graph: critic approved (score={state.get('quality_score', 0):.2f}); routing to writer_final.")
        return "writer_final"

    plan: list[str] = state.get("plan", [])
    idx: int = state.get("current_task_index", 0)

    if idx < len(plan):
        logger.info(f"Graph: more research needed; routing back to researcher (step {idx+1}/{len(plan)}).")
        return "researcher"

    # All plan steps done but critic still unsatisfied → force final
    logger.info("Graph: all plan steps done; forcing finalization.")
    return "writer_final"


def _after_researcher(state: AgentState) -> Literal["writer_draft", "critic"]:
    """
    After researcher completes one step:
      - If there are more plan steps, route back through writer_draft → critic loop
      - If all steps done, also route through the same loop (critic decides)
    In practice, always go to writer_draft first so the Critic has a report to evaluate.
    """
    return "writer_draft"


# --------------------------------------------------------------------------- #
# Graph compilation                                                            #
# --------------------------------------------------------------------------- #
def build_graph() -> "CompiledGraph":  # type: ignore[type-arg]
    """
    Construct and compile the LangGraph StateGraph.

    Node names used as route targets:
        planner, researcher, writer_draft, critic, writer_final
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner", _planner)
    graph.add_node("researcher", _researcher)
    graph.add_node("writer_draft", _writer)
    graph.add_node("critic", _critic)
    graph.add_node("writer_final", _writer)

    # Entry point
    graph.add_edge(START, "planner")

    # Planner → Researcher
    graph.add_edge("planner", "researcher")

    # Researcher → Writer (draft)
    graph.add_conditional_edges(
        "researcher",
        _after_researcher,
        {"writer_draft": "writer_draft"},
    )

    # Writer (draft) → Critic
    graph.add_edge("writer_draft", "critic")

    # Critic → conditional branch
    graph.add_conditional_edges(
        "critic",
        _should_continue_research,
        {
            "researcher": "researcher",
            "writer_final": "writer_final",
        },
    )

    # Writer (final) → END
    graph.add_edge("writer_final", END)

    compiled = graph.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return compiled


# Module-level compiled graph instance (lazy)
_compiled_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
