# AI-Research-Agent/agents/researcher.py
"""
Researcher Agent Node
---------------------
Executes one plan step per invocation, dynamically routing between:
  1. Local FAISS vector store  – preferred when documents are indexed
  2. DuckDuckGo web search     – used when local knowledge is insufficient
                                 or the vector store is empty

The node advances `current_task_index` by 1 each call so the graph
progresses through the plan sequentially.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchResults
from loguru import logger

from app.config import settings
from app.state import AgentState
from app.rag.vector_store import get_vector_store
from app.providers.base import FATAL_PROVIDER_ERRORS, Message
from app.providers.factory import get_llm


# --------------------------------------------------------------------------- #
# Prompt templates                                                             #
# --------------------------------------------------------------------------- #
_ROUTING_SYSTEM = """You are a research routing assistant.
Given a research sub-task and the available context, decide how to answer.
Return ONLY one of: "local" or "web"
- "local"  → when the sub-task is likely covered by uploaded documents
- "web"    → when real-time data, URLs, or broad general knowledge is needed
"""

_SYNTHESIS_SYSTEM = """You are a rigorous research assistant.
Synthesize the provided evidence into a dense, factual answer to the sub-task.
- Be concise but thorough (150-300 words).
- Preserve important numbers, names, and dates.
- Do NOT invent facts not present in the evidence.
- End with: SOURCES: [comma-separated list of source names]
"""


# --------------------------------------------------------------------------- #
# Singletons                                                                  #
# --------------------------------------------------------------------------- #
_search_tool: DuckDuckGoSearchResults | None = None


def _get_search() -> DuckDuckGoSearchResults:
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchResults(
            num_results=settings.MAX_SEARCH_RESULTS,
            output_format="list",
        )
    return _search_tool


# --------------------------------------------------------------------------- #
# Routing logic                                                                #
# --------------------------------------------------------------------------- #
def _decide_route(task: str, has_local_docs: bool, llm) -> str:
    """Return 'local' or 'web' based on LLM judgment + doc availability."""
    if not has_local_docs:
        return "web"

    try:
        resp = llm.generate(
            [Message("system", _ROUTING_SYSTEM), Message("user", f"Sub-task: {task}")],
            temperature=0.0,
        )
        decision = resp.strip().lower()
        return "local" if "local" in decision else "web"
    except FATAL_PROVIDER_ERRORS:
        raise
    except Exception as exc:
        logger.warning(f"Routing LLM call failed ({exc}); defaulting to 'web'.")
        return "web"


# --------------------------------------------------------------------------- #
# Local retrieval                                                              #
# --------------------------------------------------------------------------- #
def _local_search(task: str) -> list[dict]:
    """Query the FAISS index and return formatted evidence blocks."""
    vs = get_vector_store()
    if vs is None:
        return []

    try:
        docs_and_scores = vs.similarity_search_with_score(task, k=settings.RETRIEVER_TOP_K)
        blocks: list[dict] = []
        for doc, score in docs_and_scores:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", Page {page}" if page else ""
            blocks.append({
                "content": doc.page_content,
                "source": f"{source}{page_info}",
                "score": float(score),
                "type": "local",
            })
        logger.info(f"Local retrieval returned {len(blocks)} blocks.")
        return blocks
    except Exception as exc:
        logger.error(f"Local retrieval error: {exc}")
        return []


# --------------------------------------------------------------------------- #
# Web search                                                                   #
# --------------------------------------------------------------------------- #
def _web_search(task: str) -> tuple[list[dict], list[dict]]:
    """
    Run DuckDuckGo search.
    Returns (research_blocks, web_results).
    """
    search = _get_search()
    try:
        raw_results = search.run(task)
        # DuckDuckGoSearchResults with output_format="list" returns a list of dicts
        if isinstance(raw_results, str):
            import ast
            try:
                raw_results = ast.literal_eval(raw_results)
            except Exception:
                raw_results = [{"snippet": raw_results, "title": "Web Result", "link": ""}]

        web_results: list[dict] = []
        research_blocks: list[dict] = []

        for item in raw_results[:settings.MAX_SEARCH_RESULTS]:
            if isinstance(item, dict):
                title = item.get("title", "")
                link = item.get("link", item.get("href", ""))
                body = item.get("body", item.get("snippet", ""))
            else:
                title, link, body = str(item), "", ""

            web_results.append({"title": title, "href": link, "body": body})
            research_blocks.append({
                "content": f"{title}\n{body}",
                "source": link or title,
                "score": 1.0,
                "type": "web",
            })

        logger.info(f"Web search returned {len(web_results)} results.")
        return research_blocks, web_results

    except Exception as exc:
        logger.error(f"Web search error: {exc}")
        return [], []


# --------------------------------------------------------------------------- #
# Evidence synthesis                                                           #
# --------------------------------------------------------------------------- #
def _synthesize(task: str, blocks: list[dict], llm) -> str:
    """Ask the LLM to synthesize retrieved evidence into a coherent answer."""
    if not blocks:
        return f"No evidence found for: {task}"

    evidence_text = "\n\n---\n\n".join(
        f"[Source: {b['source']}]\n{b['content'][:800]}" for b in blocks
    )

    try:
        resp = llm.generate(
            [
                Message("system", _SYNTHESIS_SYSTEM),
                Message("user", (
                    f"Sub-task: {task}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    "Write your synthesized answer:"
                )),
            ],
            temperature=settings.LLM_TEMPERATURE,
        )
        return resp.strip()
    except FATAL_PROVIDER_ERRORS:
        raise
    except Exception as exc:
        logger.error(f"Synthesis LLM call failed: {exc}")
        # Return raw evidence as fallback
        return "\n".join(b["content"][:300] for b in blocks[:3])


# --------------------------------------------------------------------------- #
# Node callable                                                                #
# --------------------------------------------------------------------------- #
def researcher_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    LangGraph node: Researcher Agent.

    Reads:  state["plan"], state["current_task_index"]
    Writes: state["research_blocks"], state["web_results"],
            state["current_task_index"], state["messages"], state["citations"]
    """
    session_id = state.get("session_id", "unknown")
    plan: list[str] = state.get("plan", [])
    idx: int = state.get("current_task_index", 0)

    if idx >= len(plan):
        logger.info(f"[{session_id}] All plan steps completed.")
        return {"messages": [{"role": "researcher", "content": "✅ All research tasks completed."}]}

    task = plan[idx]
    logger.info(f"[{session_id}] Researcher executing step {idx+1}/{len(plan)}: {task}")

    llm = get_llm(config)   # fails fast (missing key / Ollama down) before any work is done

    # Determine if there are locally indexed documents
    vs = get_vector_store()
    has_local = vs is not None

    # Route
    route = _decide_route(task, has_local, llm)
    logger.info(f"[{session_id}] Route decision: {route}")

    research_blocks: list[dict] = []
    web_results: list[dict] = []

    if route == "local":
        research_blocks = _local_search(task)
        if not research_blocks:
            # Fallback to web if local returns nothing
            logger.info(f"[{session_id}] Local empty; falling back to web search.")
            research_blocks, web_results = _web_search(task)
    else:
        research_blocks, web_results = _web_search(task)

    # Synthesize answer
    synthesis = _synthesize(task, research_blocks, llm)

    # Build citations for new blocks
    existing_count = len(state.get("citations", []))
    new_citations: list[str] = []
    for i, block in enumerate(research_blocks):
        citation_num = existing_count + i + 1
        new_citations.append(f"[{citation_num}] {block['source']}")

    message_content = (
        f"**Step {idx+1}: {task}**\n\n"
        f"*Route: {'📄 Local FAISS' if route == 'local' else '🌐 Web Search'}*\n\n"
        f"{synthesis}"
    )

    return {
        "research_blocks": research_blocks,
        "web_results": web_results,
        "current_task_index": idx + 1,
        "citations": new_citations,
        "messages": [{"role": "researcher", "content": message_content}],
    }
