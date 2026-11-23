# backend/app/config.py
"""
Central configuration module for the Local AI Research Agent.
All model names, directory paths, and tunable constants live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).resolve().parent.parent   # -> backend/
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    FAISS_INDEX_DIR: Path = BASE_DIR / "data" / "faiss_index"
    REPORTS_DIR: Path = BASE_DIR / "data" / "reports"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ------------------------------------------------------------------ #
    #  Ollama / LLM                                                        #
    # ------------------------------------------------------------------ #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Primary model - switch to "llama3" if preferred
    LLM_MODEL: str = "mistral"
    # Temperature for creative/writer nodes; lower = more deterministic
    LLM_TEMPERATURE: float = 0.2
    LLM_CONTEXT_WINDOW: int = 8192
    LLM_MAX_TOKENS: int = 4096
    # Seconds to wait on a single Ollama generation before giving up.
    OLLAMA_TIMEOUT: float = 300.0

    # ------------------------------------------------------------------ #
    #  Provider selection (Local Ollama vs Gemini)                         #
    # ------------------------------------------------------------------ #
    # Only used when a request does not name a provider. A provider that is
    # selected but fails is NEVER silently replaced by the other one.
    DEFAULT_PROVIDER: Literal["ollama", "gemini"] = "ollama"
    # Optional: may also be supplied at runtime via POST /settings/gemini-key.
    # Never returned by any endpoint and never logged.
    GEMINI_API_KEY: SecretStr | None = None
    # Model names change often; override in .env as needed.
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT: float = 60.0
    # Upload hardening (applies to every upload endpoint)
    MAX_UPLOAD_MB: int = 50
    MAX_UPLOADS_PER_REQUEST: int = 20
    MAX_PDF_PAGES: int = 1000

    # ------------------------------------------------------------------ #
    #  Document intelligence                                               #
    # ------------------------------------------------------------------ #
    WORKSPACES_DIR: Path = BASE_DIR / "data" / "workspaces"
    DB_PATH: Path = BASE_DIR / "data" / "app.db"
    # "hf" = local sentence-transformers model (production). "hashing" = tiny
    # dependency-free lexical embedder for CI / offline smoke tests only.
    EMBEDDING_BACKEND: Literal["hf", "hashing"] = "hf"
    QA_TOP_K: int = 6
    # Evidence gates for document-grounded answers. These are STARTING VALUES for
    # all-MiniLM-L6-v2 cosine similarity; calibrate on your own documents with
    # `python -m scripts.calibrate_thresholds` (see README).
    RETRIEVAL_MIN_SIMILARITY: float = 0.30      # below this a chunk is not evidence
    RETRIEVAL_STRONG_SIMILARITY: float = 0.55   # above this lexical check is waived
    EVIDENCE_MIN_TERM_COVERAGE: float = 0.34    # share of question terms found in evidence
    CSV_ROWS_PER_CHUNK: int = 25

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DEVICE: str = "cpu"   # change to "cuda" if GPU is available

    # ------------------------------------------------------------------ #
    #  RAG / Chunking                                                      #
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_TOP_K: int = 6     # docs returned per retrieval call

    # ------------------------------------------------------------------ #
    #  Agent Graph                                                         #
    # ------------------------------------------------------------------ #
    MAX_ITERATIONS: int = 5      # hard ceiling to prevent infinite loops
    REFLECTION_THRESHOLD: float = 0.6  # minimum quality score to skip re-loop
    MAX_SEARCH_RESULTS: int = 5  # DuckDuckGo results per query

    # ------------------------------------------------------------------ #
    #  FastAPI                                                             #
    # ------------------------------------------------------------------ #
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    CORS_ORIGINS: list[str] = [
        "http://localhost:8501", "http://127.0.0.1:8501",   # legacy Streamlit
        "http://localhost:5173", "http://127.0.0.1:5173",   # Vite / React dev server
    ]

    # ------------------------------------------------------------------ #
    #  Streamlit                                                           #
    # ------------------------------------------------------------------ #
    STREAMLIT_PORT: int = 8501   # legacy UI only

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "10 MB"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton accessible throughout the project
settings = Settings()

# Ensure required directories exist at import time
for _dir in (
    settings.DATA_DIR,
    settings.UPLOAD_DIR,
    settings.WORKSPACES_DIR,
    settings.FAISS_INDEX_DIR,
    settings.REPORTS_DIR,
    settings.LOG_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
