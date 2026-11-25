"""
Local embeddings, independent of the selected LLM provider (Ollama / Gemini).

* "hf"      – sentence-transformers model, runs fully locally (default).
* "hashing" – dependency-free lexical embedder for CI and offline smoke tests.
              NOT a semantic model; never use it for real documents.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol

import numpy as np

from app.config import settings
from app.core.errors import EmbeddingError
from app.rag.text import STOPWORDS, stem, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    dim = 384

    def __init__(self) -> None:
        self.name = "hashing-384"

    def _vec(self, text: str) -> np.ndarray:
        toks = [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            v[int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        v = np.log1p(v)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class HFEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "The local embedding model needs `sentence-transformers` (pip install -r requirements.txt)."
            ) from exc
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}'. It downloads once on first use, "
                f"so check your internet connection or pre-download it. ({type(exc).__name__})"
            ) from exc
        self.name, self.dim = model_name, int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


_lock = threading.Lock()
_cache: dict[tuple, Embedder] = {}


def get_embedder() -> Embedder:
    key = (settings.EMBEDDING_BACKEND, settings.EMBED_MODEL, settings.EMBED_DEVICE)
    with _lock:
        if key not in _cache:
            _cache[key] = HashingEmbedder() if settings.EMBEDDING_BACKEND == "hashing" \
                else HFEmbedder(settings.EMBED_MODEL, settings.EMBED_DEVICE)
        return _cache[key]


def reset_embedder() -> None:
    global _last_error
    with _lock:
        _cache.clear()
        _last_error = None


_last_error: str | None = None


def embedding_state() -> dict:
    """For the UI: is the local embedding model loaded, or did loading fail (e.g. offline first run)?"""
    key = (settings.EMBEDDING_BACKEND, settings.EMBED_MODEL, settings.EMBED_DEVICE)
    with _lock:
        return {"loaded": key in _cache or settings.EMBEDDING_BACKEND == "hashing", "error": _last_error}


def warm_up() -> None:
    """Load (and, on first run, download) the embedding model so the first upload/question is not slow."""
    global _last_error
    from loguru import logger
    try:
        emb = get_embedder()
        _last_error = None
        logger.info(f"Embedding model ready: {emb.name} (dim {emb.dim}).")
    except EmbeddingError as exc:
        _last_error = exc.message
        logger.warning(f"Embedding model not available yet: {exc.message}")
