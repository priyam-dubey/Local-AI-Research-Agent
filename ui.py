# AI-Research-Agent/ui.py
"""
Streamlit Frontend
-------------------
Provides:
  - Sidebar: file/URL/text upload panel with FAISS index stats
  - Main panel: research query input + live agent step feed
  - Report view: rendered Markdown + PDF export button
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Preformatted,
)
from reportlab.lib import colors

from config import settings


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
API_BASE = f"http://localhost:{settings.API_PORT}"

ROLE_CONFIG = {
    "planner":    {"icon": "🗺️",  "color": "#2563EB", "label": "Planner"},
    "researcher": {"icon": "🔬",  "color": "#059669", "label": "Researcher"},
    "critic":     {"icon": "🧐",  "color": "#D97706", "label": "Critic"},
    "writer":     {"icon": "✍️",  "color": "#7C3AED", "label": "Writer"},
    "system":     {"icon": "⚙️",  "color": "#6B7280", "label": "System"},
}


# --------------------------------------------------------------------------- #
# PDF export helper                                                            #
# --------------------------------------------------------------------------- #
def _generate_pdf(report_md: str, query: str) -> bytes:
    """Convert a Markdown report to a clean PDF using reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1E40AF"),
        spaceAfter=8,
        spaceBefore=16,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=4,
    )

    story = []

    # Header
    story.append(Paragraph("Local AI Research Agent", title_style))
    story.append(Paragraph(f"Query: {query}", meta_style))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 12))

    # Parse Markdown lines (simple parser)
    for line in report_md.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], h1_style))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(f"<b>{line[4:]}</b>", body_style))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {line[2:]}", body_style))
        elif line.startswith("> "):
            quote_style = ParagraphStyle(
                "Quote", parent=body_style,
                leftIndent=20, textColor=colors.HexColor("#4B5563"),
                borderPad=4,
            )
            story.append(Paragraph(line[2:], quote_style))
        else:
            # Escape HTML special chars for reportlab
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_line, body_style))

    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# API helpers                                                                  #
# --------------------------------------------------------------------------- #
def _api_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _api_post(path: str, **kwargs) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", timeout=30, **kwargs)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# --------------------------------------------------------------------------- #
# Streamlit page config                                                        #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Local AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.agent-card {
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 4px solid;
    background: rgba(255,255,255,0.03);
}
.agent-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.7;
    margin-bottom: 4px;
}
.step-badge {
    display: inline-block;
    background: #1E40AF;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
}
.index-stat {
    background: #0F172A;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94A3B8;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar: document ingestion                                                  #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🔬 Local AI Research Agent")
    st.markdown("*Fully offline • LangGraph + Ollama*")
    st.divider()

    # ── Index status ─────────────────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    idx_status = _api_get("/index/status")
    if idx_status:
        if idx_status.get("status") == "empty":
            st.info("Index is empty. Upload documents below.")
        else:
            st.success(f"✅ {idx_status['doc_count']} chunks indexed")
            if idx_status.get("sources"):
                with st.expander("Indexed sources"):
                    for src in idx_status["sources"]:
                        st.markdown(f"<div class='index-stat'>📄 {src}</div>", unsafe_allow_html=True)

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        resp = _api_post("/index/reset")
        if resp:
            st.success("Knowledge base cleared.")
            st.rerun()

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("⬆️ Ingest Files", use_container_width=True):
        for ufile in uploaded_files:
            with st.spinner(f"Ingesting {ufile.name}..."):
                resp = httpx.post(
                    f"{API_BASE}/ingest/file",
                    files={"file": (ufile.name, ufile.getvalue(), ufile.type)},
                    timeout=60,
                )
                result = resp.json()
                if "error" in result:
                    st.error(f"❌ {ufile.name}: {result['error']}")
                else:
                    st.success(f"✅ {ufile.name}: {result['chunks_added']} chunks added")
        st.rerun()

    st.divider()

    # ── URL ingestion ─────────────────────────────────────────────────────────
    st.markdown("### 🌐 Ingest URL")
    url_input = st.text_input("Web URL", placeholder="https://example.com/article", label_visibility="collapsed")
    if st.button("⬆️ Ingest URL", use_container_width=True) and url_input:
        with st.spinner("Fetching and indexing URL..."):
            result = _api_post("/ingest/url", json={"url": url_input})
        if result and "error" not in result:
            st.success(f"✅ {result['chunks_added']} chunks added")
        elif result:
            st.error(result.get("error", "Unknown error"))
        st.rerun()

    st.divider()

    # ── Raw text ingestion ────────────────────────────────────────────────────
    st.markdown("### 📋 Paste Text")
    pasted_text = st.text_area("Raw text to ingest", height=100, label_visibility="collapsed")
    text_label = st.text_input("Source label", value="pasted_content", label_visibility="collapsed")
    if st.button("⬆️ Ingest Text", use_container_width=True) and pasted_text:
        result = _api_post("/ingest/text", json={"text": pasted_text, "source_name": text_label})
        if result and "error" not in result:
            st.success(f"✅ {result['chunks_added']} chunks added")
        st.rerun()

    st.divider()
    st.markdown(
        "<small>Model: **{}** | Embeddings: all-MiniLM-L6-v2</small>".format(settings.LLM_MODEL),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main panel                                                                   #
# --------------------------------------------------------------------------- #
st.markdown("# 🔬 Research Query")

query_col, btn_col = st.columns([5, 1])
with query_col:
    query = st.text_area(
        "Enter your research question",
        height=90,
        placeholder="e.g. What are the latest advances in quantum error correction?",
        label_visibility="collapsed",
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Research", type="primary", use_container_width=True)

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "research_done" not in st.session_state:
    st.session_state.research_done = False


# ── Start research ─────────────────────────────────────────────────────────────
if run_btn and query.strip():
    st.session_state.messages = []
    st.session_state.final_report = None
    st.session_state.research_done = False

    start_resp = _api_post("/research", json={"query": query.strip()})
    if start_resp and "session_id" in start_resp:
        st.session_state.session_id = start_resp["session_id"]
    else:
        st.error("Failed to start research session. Is the API running?")


# ── Live event stream ─────────────────────────────────────────────────────────
if st.session_state.session_id and not st.session_state.research_done:
    session_id = st.session_state.session_id

    st.markdown(f"**Session:** `{session_id}`")

    agent_container = st.container()
    status_placeholder = st.empty()

    # Poll the SSE stream synchronously in Streamlit
    with status_placeholder:
        with st.spinner("🔄 Agents working..."):
            try:
                with httpx.stream(
                    "GET",
                    f"{API_BASE}/research/{session_id}/stream",
                    timeout=300,
                ) as response:
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                            role = event.get("role", "system")
                            content = event.get("content", "")

                            if "[COMPLETE]" in content or "[ERROR]" in content:
                                st.session_state.research_done = True
                                break

                            st.session_state.messages.append({"role": role, "content": content})

                        except json.JSONDecodeError:
                            pass
            except Exception as exc:
                st.error(f"Stream error: {exc}")

    # Fetch final result
    result = _api_get(f"/research/{session_id}/result")
    if result:
        st.session_state.final_report = result.get("final_report", "")
        st.session_state.research_done = True

    status_placeholder.empty()
    st.rerun()


# ── Render agent messages ─────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("### 🤖 Agent Activity Log")
    for msg in st.session_state.messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        cfg = ROLE_CONFIG.get(role, ROLE_CONFIG["system"])

        st.markdown(
            f"""<div class="agent-card" style="border-color:{cfg['color']}">
            <div class="agent-label" style="color:{cfg['color']}">{cfg['icon']} {cfg['label']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(content)
        st.markdown("---")


# ── Final report display ──────────────────────────────────────────────────────
if st.session_state.final_report:
    st.markdown("---")
    st.markdown("## 📄 Final Research Report")

    report_tab, raw_tab = st.tabs(["📖 Rendered", "🗒️ Raw Markdown"])

    with report_tab:
        st.markdown(st.session_state.final_report)

    with raw_tab:
        st.code(st.session_state.final_report, language="markdown")

    # PDF export
    st.markdown("### 📥 Export")
    pdf_bytes = _generate_pdf(st.session_state.final_report, query or "Research Query")
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"research_report_{int(time.time())}.pdf",
        mime="application/pdf",
        type="primary",
    )

elif not st.session_state.session_id:
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#64748B">
            <div style="font-size:64px">🔬</div>
            <h3>Enter a research question to get started</h3>
            <p>The agent will plan, research, critique, and write a comprehensive report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
