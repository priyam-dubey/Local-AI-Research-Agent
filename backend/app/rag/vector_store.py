"""
Per-workspace FAISS indexes (cosine similarity via inner product on normalised vectors).

* One directory per workspace => documents of different workspaces can never
  be retrieved together (isolation by construction, not by filtering).
* Persisted as: index.faiss, vectors.npy, chunks.json, meta.json. No pickle files.
* Deleting a document rebuilds that workspace's index from the stored vectors.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.config import settings
from app.core.errors import EmbeddingError


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class WorkspaceIndex:
    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.dir = settings.WORKSPACES_DIR / workspace_id
        self.lock = threading.RLock()
        self.chunks: list[dict] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.embed_name = ""
        self.index: Optional[faiss.Index] = None
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            meta = json.loads((self.dir / "meta.json").read_text())
            self.chunks = json.loads((self.dir / "chunks.json").read_text())
            self.vectors = np.load(self.dir / "vectors.npy")
            self.embed_name = meta.get("embedding_model", "")
        except (FileNotFoundError, ValueError):
            return
        if len(self.chunks) != len(self.vectors):          # corrupted/partial write -> start clean
            self.chunks, self.vectors = [], np.zeros((0, 0), dtype=np.float32)
            return
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        if len(self.vectors) == 0:
            self.index = None
            return
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        import io
        buf = io.BytesIO()
        np.save(buf, self.vectors)
        _atomic_write(self.dir / "vectors.npy", buf.getvalue())
        _atomic_write(self.dir / "chunks.json", json.dumps(self.chunks).encode())
        _atomic_write(self.dir / "meta.json", json.dumps(
            {"embedding_model": self.embed_name, "dim": int(self.vectors.shape[1]) if len(self.vectors) else 0}).encode())
        if self.index is not None:
            faiss.write_index(self.index, str(self.dir / "index.faiss"))
        else:
            (self.dir / "index.faiss").unlink(missing_ok=True)

    def _check_model(self, name: str) -> None:
        if self.chunks and self.embed_name and self.embed_name != name:
            raise EmbeddingError(
                f"This workspace was indexed with '{self.embed_name}' but '{name}' is configured. "
                "Re-upload the documents, or restore the previous EMBED_MODEL / EMBEDDING_BACKEND.")

    # ---------------------------------------------------------- mutation
    def add(self, chunks: list[dict], vectors: np.ndarray, embed_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        with self.lock:
            self._check_model(embed_name)
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            self.vectors = vecs if len(self.vectors) == 0 else np.vstack([self.vectors, vecs])
            self.chunks.extend(chunks)
            self.embed_name = embed_name
            self._rebuild_faiss()
            self._save()

    def remove_document(self, doc_id: str) -> int:
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
                self._rebuild_faiss()
                self._save()
            return removed

    # ---------------------------------------------------------- queries
    @property
    def empty(self) -> bool:
        return not self.chunks

    def doc_ids(self) -> set[str]:
        return {c["doc_id"] for c in self.chunks}

    def doc_chunks(self, doc_id: str) -> list[dict]:
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    def search(self, qvec: np.ndarray, k: int, doc_ids: Optional[set[str]] = None,
               embed_name: Optional[str] = None) -> list[tuple[dict, float]]:
        with self.lock:
            if self.index is None or not self.chunks:
                return []
            if embed_name:
                self._check_model(embed_name)
            fetch = len(self.chunks) if doc_ids is not None else min(k, len(self.chunks))
            scores, ids = self.index.search(np.ascontiguousarray(qvec[None, :], dtype=np.float32), fetch)
            out = []
            for s, i in zip(scores[0], ids[0]):
                if i < 0:
                    continue
                c = self.chunks[i]
                if doc_ids is not None and c["doc_id"] not in doc_ids:
                    continue
                out.append((c, float(s)))
                if len(out) >= k:
                    break
            return out


_registry: dict[str, WorkspaceIndex] = {}
_reg_lock = threading.Lock()


def get_index(workspace_id: str) -> WorkspaceIndex:
    with _reg_lock:
        if workspace_id not in _registry:
            _registry[workspace_id] = WorkspaceIndex(workspace_id)
        return _registry[workspace_id]


def drop_index(workspace_id: str) -> None:
    with _reg_lock:
        _registry.pop(workspace_id, None)
    shutil.rmtree(settings.WORKSPACES_DIR / workspace_id, ignore_errors=True)


def clear_registry() -> None:
    """Forget cached in-memory indexes (tests / config changes)."""
    with _reg_lock:
        _registry.clear()
