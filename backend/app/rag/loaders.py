"""
Text extraction with page/section metadata for PDF, DOCX, TXT, Markdown, CSV.

Every loader returns `Section`s; `page` is only set when the format has real
pages (PDF). Other formats get a descriptive `location` ("Section: Methods",
"Rows 26-50") so citations remain traceable without inventing page numbers.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings
from app.core.errors import IngestionError, UnsupportedTypeError

ALLOWED_TYPES = {"pdf": "pdf", "docx": "docx", "txt": "txt", "md": "md", "markdown": "md", "csv": "csv"}


@dataclass
class Section:
    text: str
    location: str
    page: Optional[int] = None


def detect_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_TYPES:
        raise UnsupportedTypeError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted({*ALLOWED_TYPES}))}.")
    return ALLOWED_TYPES[ext]


def validate_file(path: Path, file_type: str) -> None:
    """Content sniffing – the extension alone is never trusted."""
    head = path.read_bytes()[:4096] if path.stat().st_size else b""
    if not head:
        raise IngestionError("The file is empty.")
    if file_type == "pdf" and not head.lstrip().startswith(b"%PDF"):
        raise UnsupportedTypeError("This file has a .pdf extension but is not a PDF.")
    if file_type == "docx":
        if not head.startswith(b"PK\x03\x04"):
            raise UnsupportedTypeError("This file has a .docx extension but is not a Word document.")
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" not in z.namelist():
                    raise UnsupportedTypeError("This file is not a valid .docx document.")
                if sum(i.file_size for i in z.infolist()) > 300 * 1024 * 1024:   # zip-bomb guard
                    raise IngestionError("The document expands to an unreasonable size.")
        except zipfile.BadZipFile as exc:
            raise IngestionError("The .docx file is corrupted.") from exc
    if file_type in ("txt", "md", "csv") and b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UnsupportedTypeError("This looks like a binary file, not text.")


def load_document(path: Path, file_type: str) -> list[Section]:
    validate_file(path, file_type)
    loader = {"pdf": _load_pdf, "docx": _load_docx, "txt": _load_text, "md": _load_text, "csv": _load_csv}[file_type]
    sections = [s for s in loader(path) if s.text.strip()]
    if not sections:
        raise IngestionError(
            "No extractable text was found." + (" This may be a scanned PDF; OCR is not supported." if file_type == "pdf" else ""))
    return sections


# ------------------------------------------------------------------ PDF
def _load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestionError("The PDF is password-protected.")
        out = []
        for i, page in enumerate(reader.pages[: settings.MAX_PDF_PAGES], start=1):
            out.append(Section((page.extract_text() or "").strip(), f"Page {i}", i))
        return out
    except IngestionError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise IngestionError("The PDF could not be read; it may be corrupted.") from exc


# ------------------------------------------------------------------ text helpers
def decode_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _blocks(paragraphs: list[str], limit: int = 3000) -> list[Section]:
    """Group paragraphs into ~limit-char parts labelled 'Part N' (for text without headings)."""
    out, buf, size = [], [], 0
    for p in paragraphs:
        if buf and size + len(p) > limit:
            out.append(Section("\n\n".join(buf), f"Part {len(out) + 1}"))
            buf, size = [], 0
        buf.append(p)
        size += len(p)
    if buf:
        out.append(Section("\n\n".join(buf), f"Part {len(out) + 1}"))
    return out


_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


def _load_text(path: Path) -> list[Section]:
    text = decode_bytes(path.read_bytes())
    lines, sections, heading, buf = text.splitlines(), [], None, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(body, f"Section: {heading}" if heading else ""))

    for line in lines:
        m = _MD_HEADING.match(line) if path.suffix.lower() in (".md", ".markdown") else None
        if m:
            flush()
            heading, buf = m.group(2).strip()[:80], [line]
        else:
            buf.append(line)
    flush()
    if len(sections) <= 1:            # no (useful) headings -> size-based parts
        return _blocks([p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()])
    return sections


# ------------------------------------------------------------------ DOCX
def _load_docx(path: Path) -> list[Section]:
    try:
        from docx import Document
    except ImportError as exc:
        raise IngestionError("DOCX support needs `python-docx` (pip install -r requirements.txt).") from exc
    try:
        doc = Document(str(path))
    except Exception as exc:
        raise IngestionError("The .docx file could not be read.") from exc

    sections, heading, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(body, f"Section: {heading}" if heading else ""))

    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")) and p.text.strip():
                flush()
                heading, buf = p.text.strip()[:80], [p.text.strip()]
            elif p.text.strip():
                buf.append(p.text.strip())
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    buf.append(" | ".join(cells))
    flush()
    if len(sections) <= 1:
        paras = [p for s in sections for p in s.text.split("\n") if p.strip()]
        return _blocks(paras)
    return sections


# ------------------------------------------------------------------ CSV
def _load_csv(path: Path) -> list[Section]:
    text = decode_bytes(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    csv.field_size_limit(10_000_000)
    rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 1:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [Section(" | ".join(header), "Header")]
    step, out = settings.CSV_ROWS_PER_CHUNK, []
    for start in range(0, len(body), step):
        part = body[start:start + step]
        lines = ["; ".join(f"{h.strip() or f'col{j+1}'}: {v.strip()}" for j, (h, v) in enumerate(zip(header, r)) if v.strip())
                 for r in part]
        out.append(Section("\n".join(lines), f"Rows {start + 1}-{start + len(part)}"))
    return out
