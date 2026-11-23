"""The evidence gate must separate answerable from unanswerable questions on a labelled corpus."""

import json
from pathlib import Path

from scripts.calibrate_thresholds import measure, passes, suggest
from app.config import settings

ROOT = Path(__file__).parent / "fixtures"


def test_gate_accuracy_on_labelled_questions():
    rows = measure(ROOT / "corpus", json.loads((ROOT / "questions.json").read_text()))
    cfg = (settings.RETRIEVAL_MIN_SIMILARITY, settings.RETRIEVAL_STRONG_SIMILARITY, settings.EVIDENCE_MIN_TERM_COVERAGE)
    false_accepts = [r["question"] for r in rows if r["label"] == "unanswerable" and passes(r, *cfg)]
    false_refusals = [r["question"] for r in rows if r["label"] == "answerable" and not passes(r, *cfg)]
    assert false_accepts == [], false_accepts          # hallucination risk: must be zero
    assert false_refusals == [], false_refusals


def test_similarity_alone_is_not_enough_which_is_why_the_gate_uses_two_signals():
    rows = measure(ROOT / "corpus", json.loads((ROOT / "questions.json").read_text()))
    worst_unanswerable = max(r["top"] for r in rows if r["label"] == "unanswerable")
    weakest_answerable = min(r["top"] for r in rows if r["label"] == "answerable")
    assert worst_unanswerable > weakest_answerable     # overlap in similarity => lexical coverage is required


def test_suggest_finds_a_zero_error_setting():
    rows = measure(ROOT / "corpus", json.loads((ROOT / "questions.json").read_text()))
    s = suggest(rows)
    assert s["false_accepts"] == 0 and s["false_refusals"] == 0
