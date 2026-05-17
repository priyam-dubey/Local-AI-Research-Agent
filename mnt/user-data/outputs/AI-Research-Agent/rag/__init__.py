# AI-Research-Agent/rag/__init__.py
"""RAG subsystem: document ingestion and FAISS vector store management."""

from rag.ingestion import ingest_file, ingest_url, ingest_text
from rag.vector_store import get_vector_store, add_documents, reset_vector_store

__all__ = [
    "ingest_file",
    "ingest_url",
    "ingest_text",
    "get_vector_store",
    "add_documents",
    "reset_vector_store",
]
