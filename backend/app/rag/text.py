"""Small text utilities shared by embeddings, evidence checks and answer verification."""

from __future__ import annotations

import re

STOPWORDS = frozenset("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same
she should so some such than that the their them then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours""".split())

# Words that describe the *act of asking about a document*, not its content.
GENERIC_TERMS = frozenset("""document documents paper study research text file report article author authors uploaded mentioned
discussed described explain describe summarize summary tell give main used use using list show provide please simple simply
terms word words""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(tok: str) -> str:
    for suf in ("ations", "ation", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def content_terms(text: str) -> list[str]:
    seen, out = set(), []
    for t in tokenize(text):
        if t in STOPWORDS or t in GENERIC_TERMS or len(t) < 3:
            continue
        s = stem(t)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def term_matches(term: str, token_stems: set[str]) -> bool:
    """Exact stem match, or a shared prefix of >= 5 chars ('methodology' ~ 'method')."""
    if term in token_stems:
        return True
    if len(term) < 5:
        return False
    pref = term[:5]
    return any(t.startswith(pref) or (len(t) >= 5 and pref.startswith(t[:5]) and t[:5] == pref) for t in token_stems)


def normalize_number(n: str) -> str:
    return n.replace(",", "").rstrip(".").rstrip("%")


def numbers_in(text: str) -> set[str]:
    return {normalize_number(m) for m in _NUMBER.findall(text)}
