# AI-Research-Agent/agents/planner.py
"""
Planner Agent Node
------------------
Receives the raw user query from AgentState and decomposes it into an
ordered list of atomic research sub-tasks.  The node calls the local
Ollama LLM and returns a partial AgentState update.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import settings
from state import AgentState


# --------------------------------------------------------------------------- #
# Prompt templates                                                             #
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are a meticulous research planning assistant.
Your sole job is to decompose a user's research question into a numbered list
of focused, atomic sub-tasks that a research agent can execute sequentially.

Rules:
- Output ONLY a valid JSON array of strings. No markdown fences, no prose.
- Each string is one self-contained research step (max 20 words).
- Aim for 3-7 steps depending on complexity.
- Steps must collectively cover all aspects needed for a comprehensive answer.
- Phrase each step as an action: "Find ...", "Identify ...", "Compare ...", etc.

Example output:
["Find the definition and history of quantum computing",
 "Identify current real-world applications of quantum computers",
 "Compare quantum vs classical computing performance benchmarks",
 "Summarize key challenges and future outlook"]
"""

_HUMAN_TEMPLATE = "Research question: {query}\n\nGenerate the research plan now:"


# --------------------------------------------------------------------------- #
# LLM factory (lazy singleton per process)                                    #
# --------------------------------------------------------------------------- #
_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.1,          # low temp → deterministic, structured output
            num_ctx=settings.LLM_CONTEXT_WINDOW,
        )
    return _llm


# --------------------------------------------------------------------------- #
# Parsing helper                                                               #
# --------------------------------------------------------------------------- #
def _parse_plan(raw: str) -> list[str]:
    """
    Attempt to parse a JSON array from the LLM output.
    Falls back to line-by-line extraction if JSON parse fails.
    """
    # Strip common markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try JSON array parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            return [step.strip() for step in parsed if step.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: numbered lines  "1. Do something"
    steps: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.+)", line)
        if match:
            steps.append(match.group(1).strip())
        elif line.startswith(("-", "*", "•")) and len(line) > 2:
            steps.append(line.lstrip("-*• ").strip())

    if steps:
        return steps

    # Last resort: return query as a single step
    logger.warning("Planner could not parse structured plan; using raw query.")
    return [f"Research: {raw[:200]}"]


# --------------------------------------------------------------------------- #
# Node callable                                                                #
# --------------------------------------------------------------------------- #
def planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Planner Agent.

    Reads:  state["query"]
    Writes: state["plan"], state["messages"]
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "unknown")
    logger.info(f"[{session_id}] Planner starting for query: {query[:80]}...")

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(query=query)),
        ]
        response = llm.invoke(messages)
        raw_content: str = response.content if hasattr(response, "content") else str(response)
        plan = _parse_plan(raw_content)

        logger.info(f"[{session_id}] Planner produced {len(plan)} steps: {plan}")

        return {
            "plan": plan,
            "current_task_index": 0,
            "messages": [
                {
                    "role": "planner",
                    "content": (
                        f"**Research Plan ({len(plan)} steps):**\n"
                        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
                    ),
                }
            ],
        }

    except Exception as exc:
        logger.error(f"[{session_id}] Planner failed: {exc}")
        fallback_plan = [
            f"Search for general information about: {query}",
            "Summarize the findings",
        ]
        return {
            "plan": fallback_plan,
            "current_task_index": 0,
            "error": f"Planner error (fallback plan used): {exc}",
            "messages": [
                {
                    "role": "planner",
                    "content": f"⚠️ Planner encountered an error; using simplified plan.\nError: {exc}",
                }
            ],
        }
