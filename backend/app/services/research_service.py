"""Research session lifecycle: in-memory store, graph execution, SSE event collection."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from loguru import logger

from app.core.security import redact
from app.graph import get_graph
from app.providers.base import ProviderError
from app.providers import factory
from app.providers.factory import _ERROR_BY_CODE, LLMSelection, selection_to_config
from app.state import AgentState

# session_id -> {"state", "status", "events", "final_report", "provider", "model"}
# NOTE: in-memory only; persisted history arrives with the workspace/SQLite work.
sessions: dict[str, dict[str, Any]] = {}


def collect_new_events(events: list[str], chunk: AgentState, already_sent: int) -> int:
    """
    Append only the messages that have not been emitted yet.

    `graph.astream(stream_mode="values")` yields the FULL accumulated state at
    every step, so v1 (which re-sent chunk["messages"] each time) delivered
    earlier messages again and again. Returns the new high-water mark.
    """
    messages = chunk.get("messages", []) or []
    for msg in messages[already_sent:]:
        events.append(json.dumps({"role": msg.get("role"), "content": msg.get("content")}))
    return max(already_sent, len(messages))


async def preflight(sel: LLMSelection) -> None:
    """
    Fail fast with a meaningful error if the chosen provider cannot serve the
    request (missing/invalid key, Ollama down, model not installed).
    Never falls back to another provider.
    """
    llm = factory.create_provider(sel)                       # raises MissingAPIKeyError
    health = await asyncio.to_thread(llm.health)
    if not health.available:
        exc_cls = _ERROR_BY_CODE.get(health.error_code or "", ProviderError)
        raise exc_cls(health.detail, provider=sel.provider)


async def run_session(session_id: str, initial: AgentState, sel: LLMSelection) -> None:
    session = sessions[session_id]
    sent = 0
    try:
        config = selection_to_config(sel)
        config["recursion_limit"] = 100              # plan steps + critic-added steps
        async for chunk in get_graph().astream(initial, config=config, stream_mode="values"):
            sent = collect_new_events(session["events"], chunk, sent)
            session["state"] = chunk

        final: AgentState = session["state"]
        session["final_report"] = final.get("final_report") or final.get("draft_report", "")
        session["status"] = "complete"
        logger.info(f"[{session_id}] Research complete.")
    except ProviderError as exc:
        logger.error(f"[{session_id}] Provider error ({exc.code}): {redact(exc.message)}")
        session["status"] = "error"
        session["events"].append(json.dumps({"role": "system", "content": f"Error: {redact(exc.message)}"}))
    except Exception as exc:
        logger.exception(f"[{session_id}] Graph execution error")
        session["status"] = "error"
        session["events"].append(json.dumps({"role": "system", "content": f"Error: {redact(str(exc))}"}))


def new_session(session_id: str, initial: AgentState, sel: LLMSelection) -> None:
    sessions[session_id] = {
        "state": initial, "status": "running", "final_report": None,
        "provider": sel.provider, "model": sel.model,
        "events": [json.dumps({
            "role": "system",
            "content": f"Using {'local Ollama' if sel.provider == 'ollama' else 'Gemini (online)'} for this run.",
        })],
    }
