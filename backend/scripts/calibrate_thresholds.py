"""
Calibrate the evidence gate on YOUR documents and YOUR embedding model.

    python -m scripts.calibrate_thresholds --docs ./my_docs --questions questions.json

questions.json: {"answerable": ["..."], "unanswerable": ["..."]}

For every question we measure the top retrieval similarity and the question-term
coverage, then grid-search RETRIEVAL_MIN_SIMILARITY / RETRIEVAL_STRONG_SIMILARITY /
EVIDENCE_MIN_TERM_COVERAGE. A false refusal (answerable question blocked) is
penalised less than a false accept (unanswerable question let through), because
the product goal is "never invent answers".
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

from app.config import settings
from app.rag import retrieval
from app.rag.embeddings import get_embedder
from app.rag.ingestion import chunk_sections
from app.rag.loaders import detect_type, load_document
from app.rag.vector_store import clear_registry, get_index


def measure(docs_dir: Path, questions: dict) -> list[dict]:
    """Index docs into a throw-away workspace and measure each question."""
    old = settings.WORKSPACES_DIR
    with tempfile.TemporaryDirectory() as tmp:
        settings.WORKSPACES_DIR = Path(tmp)
        clear_registry()
        try:
            emb = get_embedder()
            for i, p in enumerate(sorted(docs_dir.iterdir())):
                try:
                    chunks = chunk_sections(load_document(p, detect_type(p.name)), f"d{i}", p.name)
                except Exception:
                    continue
                get_index("calib").add(chunks, emb.embed_documents([c["text"] for c in chunks]), emb.name)
            rows = []
            for label, qs in (("answerable", questions["answerable"]), ("unanswerable", questions["unanswerable"])):
                for q in qs:
                    hits = retrieval.search("calib", q, k=settings.QA_TOP_K, min_score=-1.0)
                    rows.append({"question": q, "label": label,
                                 "top": max((h.score for h in hits), default=0.0),
                                 "cov": retrieval.term_coverage(q, [h.chunk["text"] for h in hits])})
            return rows
        finally:
            settings.WORKSPACES_DIR = old
            clear_registry()


def passes(r: dict, min_sim: float, strong: float, cov: float) -> bool:
    return r["top"] >= min_sim and (r["top"] >= strong or r["cov"] >= cov)


def suggest(rows: list[dict]) -> dict:
    sims = sorted({round(r["top"], 2) for r in rows})
    best = None
    for min_sim, strong, cov in itertools.product(
            [s for s in sims if s >= 0.0], [s for s in sims] + [1.1], [0.2, 0.25, 0.34, 0.4, 0.5, 0.6]):
        if strong < min_sim:
            continue
        fa = sum(1 for r in rows if r["label"] == "unanswerable" and passes(r, min_sim, strong, cov))
        fr = sum(1 for r in rows if r["label"] == "answerable" and not passes(r, min_sim, strong, cov))
        cost = 3 * fa + fr
        # prefer lower cost, then the most lenient (lowest) floor with margin
        key = (cost, -min_sim)
        if best is None or key < best[0]:
            best = (key, {"RETRIEVAL_MIN_SIMILARITY": min_sim, "RETRIEVAL_STRONG_SIMILARITY": strong,
                          "EVIDENCE_MIN_TERM_COVERAGE": cov, "false_accepts": fa, "false_refusals": fr})
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    a = ap.parse_args()
    rows = measure(a.docs, json.loads(a.questions.read_text()))
    for r in sorted(rows, key=lambda r: (r["label"], -r["top"])):
        print(f"{r['label']:<13} top={r['top']:.3f} cov={r['cov']:.2f}  {r['question']}")
    print("\nSuggested settings:\n" + json.dumps(suggest(rows), indent=2))


if __name__ == "__main__":
    main()
