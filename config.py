# AI-Research-Agent/config.py
"""
Central configuration module for the Local AI Research Agent.
All model names, directory paths, and tunable constants live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).parent.resolve()
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
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    # ------------------------------------------------------------------ #
    #  Streamlit                                                           #
    # ------------------------------------------------------------------ #
    STREAMLIT_PORT: int = 8501

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
    settings.FAISS_INDEX_DIR,
    settings.REPORTS_DIR,
    settings.LOG_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
