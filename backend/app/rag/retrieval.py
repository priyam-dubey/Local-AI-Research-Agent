"""
Metadata-aware retrieval + the evidence gate.

The gate is deliberately NOT just a prompt: before any LLM is called we require
(1) a similarity score above RETRIEVAL_MIN_SIMILARITY, and
(2) either a strong semantic match or enough of the question's content terms
    to actually appear in the retrieved text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.rag.embeddings import get_embedder
from app.rag.text import content_terms, stem, term_matches, tokenize
from app.rag.vector_store import get_index


@dataclass
class Hit:
    chunk: dict
    score: float


@dataclass
class Evidence:
    sufficient: bool
    reason: str
    top_score: float = 0.0
    coverage: float = 0.0


def workspace_has_docs(workspace_id: str) -> bool:
    return not get_index(workspace_id).empty


def search(workspace_id: str, query: str, k: Optional[int] = None,
           doc_ids: Optional[set[str]] = None, min_score: Optional[float] = None) -> list[Hit]:
    idx = get_index(workspace_id)
    if idx.empty or (doc_ids is not None and not doc_ids):
        return []
    emb = get_embedder()
    pairs = idx.search(emb.embed_query(query), k or settings.QA_TOP_K, doc_ids, emb.name)
    floor = settings.RETRIEVAL_MIN_SIMILARITY if min_score is None else min_score
    return [Hit(c, s) for c, s in pairs if s >= floor]


def term_coverage(question: str, texts: list[str]) -> float:
    terms = content_terms(question)
    if not terms:
        return 1.0                      # nothing lexical to verify; rely on similarity
    stems = {stem(t) for text in texts for t in tokenize(text)}
    return sum(term_matches(t, stems) for t in terms) / len(terms)


def assess_evidence(question: str, hits: list[Hit]) -> Evidence:
    if not hits:
        return Evidence(False, "no_relevant_passages")
    top = max(h.score for h in hits)
    cov = term_coverage(question, [h.chunk["text"] for h in hits])
    if top >= settings.RETRIEVAL_STRONG_SIMILARITY or cov >= settings.EVIDENCE_MIN_TERM_COVERAGE:
        return Evidence(True, "ok", top, cov)
    return Evidence(False, "question_terms_not_found_in_evidence", top, cov)
