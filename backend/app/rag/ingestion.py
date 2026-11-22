# AI-Research-Agent/rag/ingestion.py
"""
Document Ingestion Pipeline
----------------------------
Handles multi-format document loading and chunking:
  - PDF  → PyPDFLoader (page-level metadata)
  - TXT  → TextLoader
  - MD   → TextLoader (treated as plain text)
  - URL  → WebBaseLoader

All loaded documents are split with RecursiveCharacterTextSplitter
and fed into the shared FAISS vector store.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.config import settings
from app.rag.vector_store import add_documents


# --------------------------------------------------------------------------- #
# Text splitter singleton                                                      #
# --------------------------------------------------------------------------- #
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# --------------------------------------------------------------------------- #
# Loader dispatch                                                              #
# --------------------------------------------------------------------------- #
def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF file, preserving page number metadata."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # Enrich metadata
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = "pdf"
    logger.info(f"PDF loaded: {path.name} ({len(docs)} pages)")
    return docs


def _load_text(path: Path, file_type: str = "txt") -> list[Document]:
    """Load a plain text or markdown file."""
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = file_type
    logger.info(f"Text file loaded: {path.name} ({len(docs)} document(s))")
    return docs


def _load_url(url: str) -> list[Document]:
    """Fetch and parse a web page."""
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["file_type"] = "url"
    logger.info(f"URL loaded: {url} ({len(docs)} document(s))")
    return docs


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def ingest_file(file_path: str | Path) -> dict:
    """
    Ingest a local file (PDF, TXT, MD) into the vector store.

    Returns a summary dict: {source, chunks_added, file_type, error?}
    """
    path = Path(file_path)
    if not path.exists():
        return {"source": str(path), "chunks_added": 0, "error": "File not found"}

    suffix = path.suffix.lower()
    raw_docs: list[Document] = []

    try:
        if suffix == ".pdf":
            raw_docs = _load_pdf(path)
        elif suffix in (".txt", ".md", ".markdown"):
            raw_docs = _load_text(path, file_type=suffix.lstrip("."))
        else:
            return {
                "source": path.name,
                "chunks_added": 0,
                "error": f"Unsupported file type: {suffix}",
            }

        chunks = _splitter.split_documents(raw_docs)
        logger.info(f"Split into {len(chunks)} chunks from {path.name}")

        add_documents(chunks)

        return {
            "source": path.name,
            "chunks_added": len(chunks),
            "file_type": suffix.lstrip("."),
            "pages": len(raw_docs),
        }

    except Exception as exc:
        logger.error(f"Ingestion error for {path.name}: {exc}")
        return {"source": path.name, "chunks_added": 0, "error": str(exc)}


def ingest_url(url: str) -> dict:
    """
    Fetch and ingest a web URL into the vector store.

    Returns a summary dict: {source, chunks_added, error?}
    """
    try:
        raw_docs = _load_url(url)
        chunks = _splitter.split_documents(raw_docs)
        logger.info(f"Split URL {url} into {len(chunks)} chunks")
        add_documents(chunks)
        return {"source": url, "chunks_added": len(chunks), "file_type": "url"}
    except Exception as exc:
        logger.error(f"URL ingestion error for {url}: {exc}")
        return {"source": url, "chunks_added": 0, "error": str(exc)}


def ingest_text(text: str, source_name: str = "manual_input") -> dict:
    """
    Ingest raw text directly (e.g., pasted content from UI).

    Returns a summary dict: {source, chunks_added, error?}
    """
    try:
        doc = Document(
            page_content=text,
            metadata={"source": source_name, "file_type": "text"},
        )
        chunks = _splitter.split_documents([doc])
        logger.info(f"Split manual text into {len(chunks)} chunks")
        add_documents(chunks)
        return {"source": source_name, "chunks_added": len(chunks), "file_type": "text"}
    except Exception as exc:
        logger.error(f"Text ingestion error: {exc}")
        return {"source": source_name, "chunks_added": 0, "error": str(exc)}
