# AI-Research-Agent/rag/vector_store.py
"""
FAISS Vector Store Manager
--------------------------
Manages an in-memory + optionally persisted FAISS index using
HuggingFace sentence-transformers/all-MiniLM-L6-v2 embeddings.

Thread-safe via a module-level lock for concurrent FastAPI requests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger

from config import settings


# --------------------------------------------------------------------------- #
# Module-level state (process-wide singleton)                                  #
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_vector_store: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_doc_count: int = 0


# --------------------------------------------------------------------------- #
# Embeddings factory                                                           #
# --------------------------------------------------------------------------- #
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-initialize the embedding model (downloads on first call)."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBED_MODEL,
            model_kwargs={"device": settings.EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
    return _embeddings


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def get_vector_store() -> Optional[FAISS]:
    """Return the current FAISS index, or None if no documents are indexed."""
    return _vector_store


def get_doc_count() -> int:
    """Return the total number of chunks currently in the index."""
    return _doc_count


def add_documents(docs: list[Document]) -> int:
    """
    Add pre-chunked documents to the FAISS index.

    Thread-safe.  Creates the index if it doesn't exist yet.
    Returns the number of new chunks added.
    """
    global _vector_store, _doc_count

    if not docs:
        logger.warning("add_documents called with empty document list; skipping.")
        return 0

    embeddings = _get_embeddings()

    with _lock:
        try:
            if _vector_store is None:
                logger.info(f"Creating new FAISS index with {len(docs)} documents...")
                _vector_store = FAISS.from_documents(docs, embeddings)
            else:
                logger.info(f"Adding {len(docs)} documents to existing FAISS index...")
                _vector_store.add_documents(docs)

            _doc_count += len(docs)
            logger.info(f"FAISS index now contains {_doc_count} chunks.")

            # Optionally persist to disk
            _persist()
            return len(docs)

        except Exception as exc:
            logger.error(f"Failed to add documents to FAISS: {exc}")
            raise


def reset_vector_store() -> None:
    """
    Clear the in-memory FAISS index and delete persisted files.
    Used when the user wants to start fresh.
    """
    global _vector_store, _doc_count

    with _lock:
        _vector_store = None
        _doc_count = 0
        logger.info("FAISS index reset.")

        # Remove persisted index files
        index_path = settings.FAISS_INDEX_DIR
        for fname in ("index.faiss", "index.pkl"):
            fpath = index_path / fname
            if fpath.exists():
                fpath.unlink()
                logger.info(f"Deleted {fpath}")


def load_persisted_index() -> bool:
    """
    Attempt to load a previously saved FAISS index from disk.
    Returns True if successful, False otherwise.
    """
    global _vector_store, _doc_count

    index_path = settings.FAISS_INDEX_DIR
    faiss_file = index_path / "index.faiss"

    if not faiss_file.exists():
        logger.info("No persisted FAISS index found.")
        return False

    try:
        embeddings = _get_embeddings()
        with _lock:
            _vector_store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            # Estimate doc count (FAISS doesn't expose it directly)
            _doc_count = _vector_store.index.ntotal
        logger.info(f"Loaded persisted FAISS index with ~{_doc_count} vectors.")
        return True
    except Exception as exc:
        logger.error(f"Failed to load persisted FAISS index: {exc}")
        return False


def get_index_summary() -> dict:
    """Return a human-readable summary of the current index state."""
    vs = get_vector_store()
    if vs is None:
        return {"status": "empty", "doc_count": 0, "sources": []}

    # Collect unique sources from docstore
    sources: set[str] = set()
    try:
        for doc_id, doc in vs.docstore._dict.items():
            src = doc.metadata.get("source", "Unknown")
            sources.add(src)
    except Exception:
        pass

    return {
        "status": "ready",
        "doc_count": _doc_count,
        "sources": sorted(sources),
        "vector_count": vs.index.ntotal if vs else 0,
    }


# --------------------------------------------------------------------------- #
# Internal persistence helper                                                  #
# --------------------------------------------------------------------------- #
def _persist() -> None:
    """Save the FAISS index to disk (called after every add)."""
    if _vector_store is None:
        return
    try:
        _vector_store.save_local(str(settings.FAISS_INDEX_DIR))
        logger.debug(f"FAISS index persisted to {settings.FAISS_INDEX_DIR}")
    except Exception as exc:
        logger.warning(f"Could not persist FAISS index: {exc}")
