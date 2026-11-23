"""Test builders: real PDF/DOCX files, a scripted LLM, excerpt parsing."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable

from app.config import settings
from app.providers.base import BaseLLM, Message, ProviderHealth
from app.services import document_service, repo

REFUSAL = "I couldn't find this information in the uploaded document."


def make_pdf(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 740
        for line in simpleSplit(text, "Helvetica", 11, 480):
            c.setFont("Helvetica", 11)
            c.drawString(60, y, line)
            y -= 15
        c.showPage()
    c.save()
    return buf.getvalue()


def make_docx(blocks: list[tuple[str, str]], table: list[list[str]] | None = None) -> bytes:
    """blocks: ("h"|"p", text)"""
    from docx import Document
    d = Document()
    for kind, text in blocks:
        d.add_heading(text, level=1) if kind == "h" else d.add_paragraph(text)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for i, row in enumerate(table):
            for j, v in enumerate(row):
                t.cell(i, j).text = v
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def add_doc(ws_id: str, filename: str, content: bytes) -> dict:
    """Run the real processing pipeline synchronously and return the document row."""
    from app.rag.loaders import detect_type
    path = settings.UPLOAD_DIR / f"t_{filename}"
    path.write_bytes(content)
    doc = repo.create_document(ws_id, filename, detect_type(filename), len(content))
    document_service.process_document(doc, path)
    return repo.get_document(ws_id, doc["id"])


_EXC = re.compile(r'<<<EXCERPT id=(S\d+) source="([^"]*)" location="([^"]*)">>>\n(.*?)\n<<<END EXCERPT>>>', re.S)


def parse_excerpts(messages: list[Message]) -> dict[str, tuple[str, str, str]]:
    user = next(m.content for m in messages if m.role == "user")
    return {i: (src, loc, text) for i, src, loc, text in _EXC.findall(user)}


def sid_containing(messages: list[Message], needle: str) -> str:
    for i, (_, _, text) in parse_excerpts(messages).items():
        if needle.lower() in text.lower():
            return i
    raise AssertionError(f"no excerpt contains {needle!r}")


def kind_of(messages: list[Message]) -> str:
    sys = messages[0].content
    for needle, kind in (("document-grounded assistant", "qa"), ("document comparison assistant", "compare"),
                         ("rewrite a follow-up", "rewrite"), ("study questions", "questions")):
        if needle in sys:
            return kind
    return "other"


class QALLM(BaseLLM):
    """LLM whose replies come from `handler(kind, messages) -> str`; records every call."""
    provider_name, model = "fake", "fake"

    def __init__(self, handler: Callable[[str, list[Message]], str]):
        self.handler, self.calls, self.messages = handler, [], []

    def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        k = kind_of(messages)
        self.calls.append(k)
        self.messages.append(messages)
        return self.handler(k, messages)

    def stream(self, messages, **kw):
        yield self.generate(messages)

    def health(self):
        return ProviderHealth("fake", True, "Ready")
