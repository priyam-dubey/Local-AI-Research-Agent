"""PDF export of a Markdown research report (moved out of the Streamlit UI)."""

from __future__ import annotations

import io
import time
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer


def generate_pdf(report_md: str, query: str) -> bytes:
    """
    Convert a Markdown report to PDF. Every text line is XML-escaped (v1 only
    escaped plain body lines, so '&' or '<' in headings/bullets could break it).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    ss = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#1E293B"), spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor("#1E40AF"), spaceAfter=8, spaceBefore=16)
    h2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=colors.HexColor("#374151"), spaceAfter=6, spaceBefore=12)
    body = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=15, textColor=colors.HexColor("#1F2937"), spaceAfter=6)
    meta = ParagraphStyle("M", parent=ss["Normal"], fontSize=8, textColor=colors.HexColor("#6B7280"), spaceAfter=4)
    quote = ParagraphStyle("Q", parent=body, leftIndent=20, textColor=colors.HexColor("#4B5563"))

    story = [
        Paragraph("AI Research Agent", title),
        Paragraph(f"Query: {escape(query)}", meta),
        Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", meta),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 12),
    ]
    for raw in report_md.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("### "):
            story.append(Paragraph(f"<b>{escape(line[4:])}</b>", body))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), h2))
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), h1))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {escape(line[2:])}", body))
        elif line.startswith("> "):
            story.append(Paragraph(escape(line[2:]), quote))
        else:
            story.append(Paragraph(escape(line), body))
    doc.build(story)
    return buf.getvalue()
