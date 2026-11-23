"""Suggested questions generated from the ACTUAL document text, then filtered for quality."""

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from app.config import settings
from app.providers.base import BaseLLM, Message
from app.rag import retrieval
from app.rag.embeddings import get_embedder
from app.rag.vector_store import get_index

SYSTEM = ("You write study questions about a document. Use ONLY the excerpts provided. "
          "Output a JSON array of strings and nothing else.")
_BANNED = ("excerpt", "passage", "the text above", "this snippet", "the following")


def _sample_chunks(workspace_id: str, doc_ids: set[str], seed: int, limit: int = 8) -> list[dict]:
    idx = get_index(workspace_id)
    chunks = [c for d in sorted(doc_ids) for c in idx.doc_chunks(d)]
    if not chunks:
        return []
    n = min(limit, len(chunks))
    step = len(chunks) / n
    offset = seed % max(1, int(step))              # "generate more" starts from different places
    picked = list(dict.fromkeys(int(i * step + offset) % len(chunks) for i in range(n)))
    return [chunks[i] for i in picked]


def _parse(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
        except ValueError:
            pass
    return [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip() for l in raw.splitlines() if "?" in l]


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def generate_questions(workspace_id: str, doc_ids: set[str], count: int, llm: BaseLLM,
                       existing: Optional[list[str]] = None) -> list[str]:
    existing = existing or []
    sample = _sample_chunks(workspace_id, doc_ids, seed=len(existing))
    if not sample:
        return []
    excerpts = "\n\n".join(f"[{c['filename']} – {c.get('location') or 'text'}]\n{c['text'][:550]}" for c in sample)
    avoid = ("\nDo NOT repeat or rephrase these existing questions:\n- " + "\n- ".join(existing[-15:])) if existing else ""
    prompt = (f"Excerpts:\n{excerpts}\n\nWrite {count + 3} specific, self-contained questions that a reader could answer "
              "from these excerpts. Cover different aspects (objective/topic, methods, data, findings, limitations, "
              "conclusions) where the text supports them. Name the subject explicitly; no yes/no questions; never say "
              f"'the excerpt' or 'the passage'.{avoid}\nJSON array only.")
    raw = llm.generate([Message("system", SYSTEM), Message("user", prompt)], temperature=0.4, max_tokens=700)

    emb = get_embedder()
    kept: list[str] = []
    kept_vecs = [emb.embed_query(q) for q in existing]
    seen = {_norm(q) for q in existing}
    for q in _parse(raw):
        q = " ".join(q.split())
        if not (15 <= len(q) <= 220) or not q.endswith("?") or any(b in q.lower() for b in _BANNED):
            continue
        if _norm(q) in seen:
            continue
        v = emb.embed_query(q)
        if any(float(np.dot(v, o)) > 0.85 for o in kept_vecs):                 # near-duplicate
            continue
        hits = retrieval.search(workspace_id, q, k=settings.QA_TOP_K, doc_ids=doc_ids)
        if not retrieval.assess_evidence(q, hits).sufficient:                  # not answerable from the document
            continue
        kept.append(q)
        kept_vecs.append(v)
        seen.add(_norm(q))
        if len(kept) >= count:
            break
    return kept
