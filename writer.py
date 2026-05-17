# AI-Research-Agent/agents/writer.py
"""
Writer Agent Node
-----------------
Synthesizes all accumulated research_blocks and plan steps into a
polished, citation-rich Markdown report.  This node is called:
  - After each Researcher pass (to produce a `draft_report`)
  - After Critic approves (to produce the `final_report`)
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import settings
from state import AgentState


# --------------------------------------------------------------------------- #
# Prompt templates                                                             #
# --------------------------------------------------------------------------- #
_WRITER_SYSTEM = """You are an expert technical writer specializing in comprehensive research reports.

Your output must be a structured Markdown report with:
1. A clear H1 title derived from the query
2. An executive summary (2-3 sentences)
3. One H2 section per research plan step, with detailed prose (not bullets)
4. Inline citations using numbered brackets [N] wherever a fact is used
5. A "## References" section at the end listing all citations

Style guidelines:
- Professional, neutral, analytical tone
- Paragraph-form prose, not bullet lists (except for references)
- Include specific data points, names, and dates from the evidence
- Do NOT invent information — only use what appears in the evidence
- Minimum 400 words for the body
"""

_WRITER_HUMAN = """Query: {query}

Research Plan:
{plan}

Accumulated Evidence ({block_count} blocks):
{evidence}

Citations Available:
{citations}

Critic Feedback (incorporate improvements):
{critic_feedback}

Write the comprehensive Markdown report now:"""


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
            temperature=0.3,   # slightly higher for natural prose
            num_ctx=settings.LLM_CONTEXT_WINDOW,
            num_predict=settings.LLM_MAX_TOKENS,
        )
    return _llm


# --------------------------------------------------------------------------- #
# Evidence formatter                                                           #
# --------------------------------------------------------------------------- #
def _format_evidence(blocks: list[dict], citations: list[str]) -> str:
    """Combine research blocks with their citation numbers."""
    lines: list[str] = []
    for i, block in enumerate(blocks):
        citation_num = i + 1
        citation_label = citations[i] if i < len(citations) else f"[{citation_num}] Unknown"
        snippet = block.get("content", "")[:600]
        lines.append(f"{citation_label}\n{snippet}\n")
    return "\n---\n".join(lines) if lines else "No evidence available."


# --------------------------------------------------------------------------- #
# Fallback report generator (no LLM)                                          #
# --------------------------------------------------------------------------- #
def _build_fallback_report(
    query: str,
    plan: list[str],
    research_blocks: list[dict],
    citations: list[str],
) -> str:
    """Construct a minimal report from raw evidence without LLM synthesis."""
    sections: list[str] = [f"# Research Report: {query}\n"]
    sections.append("> *Auto-generated from raw evidence due to LLM synthesis error.*\n")

    for i, step in enumerate(plan):
        sections.append(f"\n## {i+1}. {step}\n")
        # Find blocks collected around this step (rough heuristic)
        relevant = research_blocks[i::len(plan)] if plan else research_blocks
        for block in relevant[:2]:
            src = block.get("source", "Unknown")
            content = block.get("content", "")[:400]
            citation_num = research_blocks.index(block) + 1 if block in research_blocks else "?"
            sections.append(f"{content} [{citation_num}]\n\n*Source: {src}*\n")

    sections.append("\n## References\n")
    for c in citations:
        sections.append(f"- {c}\n")

    return "\n".join(sections)


# --------------------------------------------------------------------------- #
# Node callable                                                                #
# --------------------------------------------------------------------------- #
def writer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Writer Agent.

    Reads:  state["query"], state["plan"], state["research_blocks"],
            state["citations"], state["critic_feedback"], state["needs_more_research"]
    Writes: state["draft_report"] | state["final_report"], state["messages"]
    """
    session_id = state.get("session_id", "unknown")
    query: str = state.get("query", "")
    plan: list[str] = state.get("plan", [])
    research_blocks: list[dict] = state.get("research_blocks", [])
    citations: list[str] = state.get("citations", [])
    critic_feedback: str = state.get("critic_feedback", "None yet.")
    needs_more: bool = state.get("needs_more_research", True)
    iteration: int = state.get("iteration", 0)

    is_final = not needs_more

    logger.info(
        f"[{session_id}] Writer generating {'FINAL' if is_final else 'DRAFT'} report "
        f"(iteration {iteration}, {len(research_blocks)} blocks)..."
    )

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    evidence_text = _format_evidence(research_blocks, citations)
    citations_text = "\n".join(citations) if citations else "No citations yet."

    prompt = _WRITER_HUMAN.format(
        query=query,
        plan=plan_text,
        block_count=len(research_blocks),
        evidence=evidence_text[:4000],   # hard cap to stay within context window
        citations=citations_text,
        critic_feedback=critic_feedback,
    )

    try:
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=_WRITER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        report: str = resp.content.strip()

        # Ensure references section is appended if LLM omitted it
        if "## References" not in report and citations:
            ref_section = "\n\n## References\n" + "\n".join(f"- {c}" for c in citations)
            report += ref_section

    except Exception as exc:
        logger.error(f"[{session_id}] Writer LLM call failed: {exc}. Using fallback report.")
        report = _build_fallback_report(query, plan, research_blocks, citations)

    label = "final_report" if is_final else "draft_report"
    status_emoji = "📝" if not is_final else "📄"
    word_count = len(report.split())

    message_content = (
        f"{status_emoji} **{'Final Report' if is_final else 'Draft Report'} Generated**\n\n"
        f"*{word_count} words • {len(research_blocks)} evidence blocks • {len(citations)} citations*\n\n"
        f"```markdown\n{report[:500]}...\n```\n*(truncated — full report available for export)*"
    )

    return {
        label: report,
        "messages": [{"role": "writer", "content": message_content}],
    }
