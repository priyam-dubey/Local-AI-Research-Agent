# AI-Research-Agent/state.py
"""
Defines the shared AgentState TypedDict that flows through every node
in the LangGraph StateGraph. All fields are explicitly typed so that
LangGraph can correctly merge partial updates returned by each node.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


# --------------------------------------------------------------------------- #
# Helper reducer: append-only list accumulator used by LangGraph               #
# --------------------------------------------------------------------------- #
def _append(existing: list, update: list) -> list:
    """Merge reducer that extends the existing list with new items."""
    return existing + update


# --------------------------------------------------------------------------- #
# Core State Schema                                                            #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    """
    Shared state dictionary passed between every LangGraph node.

    Fields
    ------
    query : str
        The original user question / research request.
    session_id : str
        Unique identifier for the current research session.
    iteration : int
        Current loop count; compared against MAX_ITERATIONS guard.
    plan : list[str]
        Ordered list of atomic sub-tasks produced by the Planner agent.
    current_task_index : int
        Pointer into `plan` indicating which step the Researcher is executing.
    research_blocks : Annotated[list[dict], _append]
        Accumulated evidence chunks. Each dict contains:
          - "content": str  – the retrieved text passage
          - "source": str   – origin (filename, URL, …)
          - "score": float  – retrieval relevance score (FAISS distance)
    web_results : Annotated[list[dict], _append]
        Raw DuckDuckGo search results (title, href, body).
    draft_report : str
        Intermediate markdown report composed by the Writer after each loop.
    final_report : str
        Polished, citation-enriched markdown report ready for export.
    citations : Annotated[list[str], _append]
        Ordered citation strings like "[1] Source: doc.pdf, Page 2".
    critic_feedback : str
        Textual feedback from the Critic/Reflection agent.
    quality_score : float
        0-1 score assigned by the Critic (≥ REFLECTION_THRESHOLD → done).
    needs_more_research : bool
        Flag set by Critic when retrieved evidence is insufficient.
    error : Optional[str]
        Captures any runtime error message for graceful degradation.
    messages : Annotated[list[dict], _append]
        Chronological log of agent messages for UI display:
          {"role": "planner"|"researcher"|"critic"|"writer", "content": str}
    metadata : dict[str, Any]
        Catch-all for auxiliary data (timestamps, token counts, etc.).
    """

    # -- Identity & control flow --
    query: str
    session_id: str
    iteration: int
    current_task_index: int

    # -- Planning --
    plan: list[str]

    # -- Research --
    research_blocks: Annotated[list[dict], _append]
    web_results: Annotated[list[dict], _append]

    # -- Writing & citation --
    draft_report: str
    final_report: str
    citations: Annotated[list[str], _append]

    # -- Reflection / Critic --
    critic_feedback: str
    quality_score: float
    needs_more_research: bool

    # -- Error handling --
    error: Optional[str]

    # -- UI feed --
    messages: Annotated[list[dict], _append]

    # -- Miscellaneous --
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Factory helper                                                               #
# --------------------------------------------------------------------------- #
def initial_state(query: str, session_id: str) -> AgentState:
    """Return a fully-initialized AgentState for a new research session."""
    return AgentState(
        query=query,
        session_id=session_id,
        iteration=0,
        current_task_index=0,
        plan=[],
        research_blocks=[],
        web_results=[],
        draft_report="",
        final_report="",
        citations=[],
        critic_feedback="",
        quality_score=0.0,
        needs_more_research=True,
        error=None,
        messages=[],
        metadata={},
    )
