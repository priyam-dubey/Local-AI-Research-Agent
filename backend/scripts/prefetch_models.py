"""
Download / verify the local embedding model once, so the first upload is instant and
you can go offline afterwards.

    python -m scripts.prefetch_models
"""

from app.config import settings
from app.core.errors import EmbeddingError
from app.rag.embeddings import get_embedder


def main() -> int:
    print(f"Embedding backend: {settings.EMBEDDING_BACKEND}  model: {settings.EMBED_MODEL}")
    try:
        emb = get_embedder()
        vec = emb.embed_query("hello world")
    except EmbeddingError as exc:
        print(f"FAILED: {exc.message}")
        return 1
    print(f"OK: {emb.name}, dimension {len(vec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
