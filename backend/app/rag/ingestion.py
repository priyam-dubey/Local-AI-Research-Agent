"""Chunking: sections -> retrieval chunks that keep filename / page / location metadata."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.loaders import Section


def chunk_sections(sections: list[Section], doc_id: str, filename: str) -> list[dict]:
    """Chunks never cross a page/section boundary, so each citation points at one place."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks: list[dict] = []
    for sec in sections:
        for piece in splitter.split_text(sec.text):
            piece = piece.strip()
            if len(piece) < 20 and len(sections) > 1:      # drop stray fragments
                continue
            if not piece:
                continue
            n = len(chunks)
            chunks.append({"chunk_id": f"{doc_id}-{n}", "doc_id": doc_id, "filename": filename,
                           "page": sec.page, "location": sec.location, "ordinal": n, "text": piece})
    return chunks
