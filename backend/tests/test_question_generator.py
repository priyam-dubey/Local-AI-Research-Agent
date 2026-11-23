import json

import pytest

from app.rag import retrieval
from app.rag.question_generator import generate_questions
from app.services import repo
from tests.helpers import QALLM, add_doc

PAPER = open("tests/fixtures/corpus/ml_paper.md", "rb").read()


@pytest.fixture
def doc():
    ws = repo.create_workspace("q")["id"]
    return ws, add_doc(ws, "ml_paper.md", PAPER)


GOOD = ["What dataset was used in the experiments?", "Which optimizer and learning rate were used to train the network?",
        "What accuracy did the convolutional network reach on the test images?",
        "What limitations does the study mention about noise and rotation?",
        "Which baseline method was compared against the network?"]


def gen(ws, doc, replies, existing=None, count=5):
    llm = QALLM(lambda k, m: replies.pop(0))
    return generate_questions(ws, {doc["id"]}, count, llm, existing or []), llm


def test_questions_come_from_the_document_and_are_answerable(doc):
    ws, d = doc
    qs, llm = gen(ws, d, [json.dumps(GOOD)])
    assert set(qs) == set(GOOD) and llm.calls == ["questions"]
    excerpt_prompt = llm.messages[0][1].content
    assert "MNIST" in excerpt_prompt                                # real document text, not a template


def test_filters_duplicates_banned_phrases_non_questions_and_unanswerable(doc):
    ws, d = doc
    noisy = [GOOD[0], GOOD[0].upper(), "What dataset was used in the experiments", "Summarise the excerpt above?",
             "What is the capital city of Australia?", "What dataset was used in these experiments?", GOOD[1], "Why?"]
    qs, _ = gen(ws, d, [json.dumps(noisy)])
    assert qs[0] == GOOD[0] and GOOD[1] in qs
    assert len(qs) == 2                                             # dup / near-dup / banned / unanswerable / short removed


def test_generate_more_never_repeats_existing_questions(doc):
    ws, d = doc
    first, _ = gen(ws, d, [json.dumps(GOOD[:3])], count=3)
    repo.add_questions(ws, d["id"], first)
    more, llm = gen(ws, d, [json.dumps(GOOD[:3] + GOOD[3:])], existing=repo.list_questions(ws, d["id"]))
    assert set(more) == set(GOOD[3:]) and not set(more) & set(first)
    assert "Do NOT repeat" in llm.messages[0][1].content


def test_tolerates_non_json_model_output(doc):
    ws, d = doc
    qs, _ = gen(ws, d, ["1. " + GOOD[0] + "\n2. " + GOOD[1] + "\nSome chatter without question"])
    assert qs == GOOD[:2]


def test_empty_workspace_returns_nothing():
    ws = repo.create_workspace("empty")["id"]
    assert generate_questions(ws, set(), 5, QALLM(lambda k, m: "[]"), []) == []
