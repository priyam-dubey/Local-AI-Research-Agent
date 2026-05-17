# AI-Research-Agent/agents/critic.py
"""
Critic / Reflection Agent Node
-------------------------------
Evaluates the accumulated research blocks and the draft report for:
  - Factual grounding (no hallucinated facts)
  - Coverage completeness relative to the original plan
  - Identifies specific gaps and generates actionable feedback

Assigns a numeric quality_score (0.0–1.0) and sets `needs_more_research`
to trigger a re-loop or allow the Writer to finalize.
"""

from __future__ import annotations

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
_CRITIC_SYSTEM = """You are a rigorous fact-checking and quality-assurance critic.
Your task is to evaluate a research draft against the original plan and evidence.

Evaluation criteria:
1. COVERAGE  – Does the draft address every step in the research plan?
2. GROUNDING – Are all claims traceable to the provided evidence? No fabrications.
3. DEPTH     – Is the information sufficiently detailed for a comprehensive report?
4. GAPS      – List specific missing topics or under-explained areas.

Respond in EXACTLY this format (fill in the brackets):
SCORE: [float between 0.0 and 1.0]
VERDICT: [PASS|NEEDS_MORE_RESEARCH]
FEEDBACK: [2-5 sentences of specific, actionable critique]
GAPS: [bullet list of missing topics, or "None" if none]
"""

_CRITIC_HUMAN = """Original Query: {query}

Research Plan:
{plan}

Evidence Blocks ({block_count} retrieved):
{evidence_summary}

Draft Report:
{draft}

Evaluate now:"""


# --------------------------------------------------------------------------- #
# LLM singleton                                                                #
# --------------------------------------------------------------------------- #
_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.1,
            num_ctx=settings.LLM_CONTEXT_WINDOW,
        )
    return _llm


# --------------------------------------------------------------------------- #
# Response parser                                                              #
# --------------------------------------------------------------------------- #
def _parse_critic_response(raw: str) -> dict:
    """Extract score, verdict, feedback, and gaps from the LLM response."""
    result = {
        "quality_score": 0.5,
        "needs_more_research": True,
        "critic_feedback": raw,
    }

    # Score
    score_match = re.search(r"SCORE:\s*([\d.]+)", raw)
    if score_match:
        try:
            result["quality_score"] = max(0.0, min(1.0, float(score_match.group(1))))
        except ValueError:
            pass

    # Verdict
    verdict_match = re.search(r"VERDICT:\s*(PASS|NEEDS_MORE_RESEARCH)", raw, re.IGNORECASE)
    if verdict_match:
        result["needs_more_research"] = verdict_match.group(1).upper() != "PASS"

    # Feedback
    feedback_match = re.search(r"FEEDBACK:\s*(.+?)(?=GAPS:|$)", raw, re.DOTALL)
    if feedback_match:
        result["critic_feedback"] = feedback_match.group(1).strip()

    # Override verdict based on score threshold for consistency
    if result["quality_score"] >= settings.REFLECTION_THRESHOLD:
        result["needs_more_research"] = False

    return result


# --------------------------------------------------------------------------- #
# Node callable                                                                #
# --------------------------------------------------------------------------- #
def critic_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Critic / Reflection Agent.

    Reads:  state["query"], state["plan"], state["research_blocks"],
            state["draft_report"], state["iteration"]
    Writes: state["quality_score"], state["needs_more_research"],
            state["critic_feedback"], state["messages"]
    """
    session_id = state.get("session_id", "unknown")
    iteration: int = state.get("iteration", 0)
    query: str = state.get("query", "")
    plan: list[str] = state.get("plan", [])
    research_blocks: list[dict] = state.get("research_blocks", [])
    draft_report: str = state.get("draft_report", "")

    logger.info(f"[{session_id}] Critic evaluating iteration {iteration}...")

    # Hard iteration ceiling — force termination regardless of quality
    if iteration >= settings.MAX_ITERATIONS:
        logger.warning(f"[{session_id}] MAX_ITERATIONS ({settings.MAX_ITERATIONS}) reached. Forcing completion.")
        return {
            "quality_score": 1.0,
            "needs_more_research": False,
            "critic_feedback": f"Maximum iterations ({settings.MAX_ITERATIONS}) reached. Proceeding with available research.",
            "messages": [
                {
                    "role": "critic",
                    "content": f"⚠️ Iteration limit reached ({settings.MAX_ITERATIONS}). Finalizing report with current data.",
                }
            ],
        }

    # Build concise evidence summary (avoid token overflow)
    evidence_summary = ""
    for i, block in enumerate(research_blocks[:10]):  # cap at 10 blocks for prompt
        src = block.get("source", "Unknown")
        snippet = block.get("content", "")[:200]
        evidence_summary += f"\n[{i+1}] {src}: {snippet}..."

    if not evidence_summary:
        evidence_summary = "No evidence blocks retrieved yet."

    plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    draft_preview = draft_report[:1500] if draft_report else "No draft generated yet."

    prompt = _CRITIC_HUMAN.format(
        query=query,
        plan=plan_text,
        block_count=len(research_blocks),
        evidence_summary=evidence_summary,
        draft=draft_preview,
    )

    try:
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw_response = resp.content.strip()
        parsed = _parse_critic_response(raw_response)

        logger.info(
            f"[{session_id}] Critic score={parsed['quality_score']:.2f}, "
            f"needs_more={parsed['needs_more_research']}"
        )

        score_emoji = "✅" if parsed["quality_score"] >= settings.REFLECTION_THRESHOLD else "🔄"
        message_content = (
            f"{score_emoji} **Critic Evaluation (Iteration {iteration + 1})**\n\n"
            f"**Quality Score:** {parsed['quality_score']:.2f} / 1.00\n\n"
            f"**Feedback:** {parsed['critic_feedback']}\n\n"
            f"**Decision:** {'Proceed to finalization' if not parsed['needs_more_research'] else 'Request additional research'}"
        )

        return {
            "quality_score": parsed["quality_score"],
            "needs_more_research": parsed["needs_more_research"],
            "critic_feedback": parsed["critic_feedback"],
            "iteration": iteration + 1,
            "messages": [{"role": "critic", "content": message_content}],
        }

    except Exception as exc:
        logger.error(f"[{session_id}] Critic node error: {exc}")
        # Safe default: proceed if we have any research at all
        has_data = len(research_blocks) > 0
        return {
            "quality_score": 0.7 if has_data else 0.0,
            "needs_more_research": not has_data,
            "critic_feedback": f"Critic evaluation failed ({exc}). {'Proceeding with available data.' if has_data else 'No data collected.'}",
            "iteration": iteration + 1,
            "error": str(exc),
            "messages": [
                {
                    "role": "critic",
                    "content": f"⚠️ Critic error: {exc}",
                }
            ],
        }
