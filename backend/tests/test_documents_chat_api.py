"""Full workflows through the HTTP API: upload -> process -> suggest -> chat -> follow-up -> compare -> delete."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.providers import factory
from app.rag.document_qa import COMPARE_REFUSAL, REFUSAL
from app.services import repo
from tests.helpers import QALLM, make_docx, make_pdf, sid_containing

client = TestClient(app)
PAPER = open("tests/fixtures/corpus/ml_paper.md", "rb").read()
CIFAR = open("tests/fixtures/corpus/cifar_paper.md", "rb").read()
REPORT = make_docx([("h", "Budget"), ("p", "The approved budget for Project Phoenix is 250,000 dollars."),
                    ("h", "Timeline"), ("p", "The beta release is scheduled for 14 March.")])


def smart(kind, msgs):
    u = msgs[-1].content if kind != "qa" else msgs[1].content
    if kind == "rewrite":
        return "What are the limitations of the convolutional network study about noise and rotation?"
    if kind == "questions":
        return json.dumps(["What dataset was used in the experiments?", "What optimizer and learning rate were used to train the network?"])
    if kind == "compare":
        a, b = sid_containing(msgs, "MNIST"), sid_containing(msgs, "CIFAR-10")
        return (f"## Similarities\n- Both train neural networks. [{a}] [{b}]\n## Differences\n- ml_paper.md uses MNIST while cifar_paper.md uses CIFAR-10. [{a}] [{b}]")
    q = u.split("QUESTION:")[-1].lower()
    for topic, key, ans in (("dataset", "MNIST dataset", "The research used the MNIST dataset."),
                            ("optimizer", "Adam optimizer", "It used the Adam optimizer with a learning rate of 0.001."),
                            ("limitation", "noise, rotation", "Robustness to noise and rotation was not evaluated."),
                            ("budget", "budget", "The approved budget is 250,000 dollars.")):
        if topic in q:
            try:
                return f"{ans} [{sid_containing(msgs, key)}]"
            except AssertionError:
                continue
    return REFUSAL


@pytest.fixture
def llm(monkeypatch):
    l = QALLM(smart)
    monkeypatch.setattr(factory, "create_provider", lambda sel: l)
    return l


@pytest.fixture
def ws():
    return client.post("/workspaces", json={"name": "Research"}).json()["id"]


def upload(ws, *files):
    return client.post(f"/workspaces/{ws}/documents", files=[("files", f) for f in files])


def test_multi_file_upload_processes_and_cleans_up(ws):
    r = upload(ws, ("ml_paper.md", PAPER, "text/markdown"), ("report.docx", REPORT, "application/octet-stream"),
               ("paper.pdf", make_pdf(["Page one about cats.", "Page two about dogs."]), "application/pdf"))
    assert r.status_code == 202 and len(r.json()["documents"]) == 3 and r.json()["rejected"] == []
    docs = client.get(f"/workspaces/{ws}/documents").json()
    assert {d["status"] for d in docs} == {"ready"} and all(d["chunks"] > 0 for d in docs)
    assert {d["file_type"] for d in docs} == {"md", "docx", "pdf"} and all(d["size_bytes"] > 0 for d in docs)
    assert list(settings.UPLOAD_DIR.iterdir()) == []                    # temp files removed


def test_bad_files_are_reported_per_file_without_failing_the_batch(ws):
    r = upload(ws, ("ok.md", PAPER, "text/markdown"), ("evil.exe", b"MZ", "application/octet-stream"),
               ("fake.pdf", b"not really a pdf", "application/pdf"))
    body = r.json()
    assert r.status_code == 202 and [d["filename"] for d in body["documents"]] == ["ok.md"]
    assert {x["filename"]: x["code"] for x in body["rejected"]} == {"evil.exe": "unsupported_file_type", "fake.pdf": "unsupported_file_type"}


def test_scanned_pdf_ends_up_failed_with_a_reason(ws):
    upload(ws, ("scan.pdf", make_pdf([" "]), "application/pdf"))
    d = client.get(f"/workspaces/{ws}/documents").json()[0]
    assert d["status"] == "failed" and "scanned" in d["error"]


def test_too_many_files_and_size_limit(ws, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOADS_PER_REQUEST", 1)
    assert upload(ws, ("a.md", PAPER, "x"), ("b.md", PAPER, "x")).status_code == 400
    monkeypatch.setattr(settings, "MAX_UPLOADS_PER_REQUEST", 20)
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    assert upload(ws, ("a.md", PAPER, "x")).json()["rejected"][0]["code"] == "file_too_large"
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_delete_document_removes_it_from_the_index(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    doc = client.get(f"/workspaces/{ws}/documents").json()[0]
    assert client.delete(f"/workspaces/{ws}/documents/{doc['id']}").status_code == 200
    assert client.get(f"/workspaces/{ws}/documents").json() == []
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "What dataset was used?"})
    assert r.status_code == 409                                            # nothing left to search
    from app.rag.vector_store import get_index
    assert get_index(ws).empty


def test_questions_are_generated_stored_and_extendable(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    doc = client.get(f"/workspaces/{ws}/documents").json()[0]
    r = client.post(f"/workspaces/{ws}/questions", json={"document_id": doc["id"], "count": 5, "provider": "ollama"})
    assert r.status_code == 200 and len(r.json()["questions"]) == 2
    stored = client.get(f"/workspaces/{ws}/questions", params={"document_id": doc["id"]}).json()["questions"]
    assert stored == r.json()["questions"]
    again = client.post(f"/workspaces/{ws}/questions", json={"document_id": doc["id"]}).json()
    assert again["questions"] == []                                        # model repeated itself -> nothing duplicated


def test_grounded_chat_with_citations_and_followup_conversation(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    doc = client.get(f"/workspaces/{ws}/documents").json()[0]
    scope = {"mode": "document", "document_ids": [doc["id"]]}
    r1 = client.post(f"/workspaces/{ws}/chat", json={"question": "What dataset of handwritten digit images was used in this research?", "scope": scope}).json()
    assert r1["grounded"] and not r1["refused"] and r1["sources"][0]["filename"] == "ml_paper.md"
    assert r1["sources"][0]["location"] == "Section: Dataset" and r1["sources"][0]["page"] is None   # md has no pages: none invented
    r2 = client.post(f"/workspaces/{ws}/chat", json={"question": "What were its limitations?", "scope": scope,
                                                    "conversation_id": r1["conversation_id"]}).json()
    assert r2["conversation_id"] == r1["conversation_id"] and "limitations" in r2["standalone_question"]
    assert not r2["refused"] and r2["sources"][0]["location"] == "Section: Limitations"
    conv = client.get(f"/workspaces/{ws}/conversations/{r1['conversation_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["meta"]["sources"][0]["filename"] == "ml_paper.md"


def test_unknown_answer_is_refused_over_http_without_calling_the_model(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "Who won the football world cup in 2018?"}).json()
    assert r["refused"] and r["answer"] == REFUSAL and r["sources"] == [] and llm.calls == []


def test_chat_stream_emits_stages_then_a_verified_final_event(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    events = []
    with client.stream("POST", f"/workspaces/{ws}/chat/stream", json={"question": "What dataset of handwritten digit images was used in this research?"}) as r:
        name = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[5:])))
    kinds = [k for k, _ in events]
    assert kinds[:3] == ["stage", "stage", "stage"] and kinds[-1] == "final"
    assert [d["stage"] for k, d in events if k == "stage"] == ["retrieving", "generating", "verifying"]
    assert events[-1][1]["sources"][0]["filename"] == "ml_paper.md" and "MNIST" in events[-1][1]["answer"]


def test_multi_document_scopes_and_comparison(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"), ("cifar_paper.md", CIFAR, "x"), ("report.docx", REPORT, "x"))
    docs = {d["filename"]: d["id"] for d in client.get(f"/workspaces/{ws}/documents").json()}
    # specific document
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "What is the approved budget for the project?",
                    "scope": {"mode": "document", "document_ids": [docs["report.docx"]]}}).json()
    assert r["sources"][0]["filename"] == "report.docx"
    # the same question restricted to a different document is refused – no cross-document leakage
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "What is the approved budget for the project?",
                    "scope": {"mode": "document", "document_ids": [docs["ml_paper.md"]]}}).json()
    assert r["refused"]
    # across all documents
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "What is the approved budget for the project?", "scope": {"mode": "all"}}).json()
    assert r["sources"][0]["filename"] == "report.docx"
    # compare two
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "Compare these documents.",
                    "scope": {"mode": "compare", "document_ids": [docs["ml_paper.md"], docs["cifar_paper.md"]]}}).json()
    assert {s["filename"] for s in r["sources"]} == {"ml_paper.md", "cifar_paper.md"} and "## Differences" in r["answer"]
    assert r["mode"] == "compare"


def test_scope_validation(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    d = client.get(f"/workspaces/{ws}/documents").json()[0]["id"]
    q = {"question": "What dataset was used?"}
    assert client.post(f"/workspaces/{ws}/chat", json={**q, "scope": {"mode": "compare", "document_ids": [d]}}).status_code == 400
    assert client.post(f"/workspaces/{ws}/chat", json={**q, "scope": {"mode": "document", "document_ids": []}}).status_code == 400
    assert client.post(f"/workspaces/{ws}/chat", json={**q, "scope": {"mode": "document", "document_ids": ["nope"]}}).status_code == 404
    assert client.post(f"/workspaces/{ws}/chat", json={"question": "   "}).status_code == 422
    assert client.post(f"/workspaces/{ws}/chat", json={**q, "conversation_id": "missing"}).status_code == 404


def test_document_ids_from_another_workspace_are_rejected(ws, llm):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    d = client.get(f"/workspaces/{ws}/documents").json()[0]["id"]
    other = client.post("/workspaces", json={"name": "Other"}).json()["id"]
    r = client.post(f"/workspaces/{other}/chat", json={"question": "What dataset was used?", "scope": {"mode": "document", "document_ids": [d]}})
    assert r.status_code == 404
    assert client.get(f"/workspaces/{other}/documents/{d}").status_code == 404
    assert client.delete(f"/workspaces/{other}/documents/{d}").status_code == 404


def test_gemini_chat_without_key_fails_clearly_and_never_falls_back(ws):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    r = client.post(f"/workspaces/{ws}/chat", json={"question": "What dataset was used?", "provider": "gemini"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_api_key"


def test_chat_uses_the_provider_the_user_selected(ws, monkeypatch):
    upload(ws, ("ml_paper.md", PAPER, "x"))
    seen = []
    monkeypatch.setattr(factory, "create_provider", lambda sel: (seen.append(sel), QALLM(smart))[1])
    client.post(f"/workspaces/{ws}/chat", json={"question": "What dataset was used?", "provider": "gemini", "model": "m1"})
    client.post(f"/workspaces/{ws}/chat", json={"question": "What dataset was used?", "provider": "ollama"})
    assert [(s.provider, s.model) for s in seen] == [("gemini", "m1"), ("ollama", None)]


def test_workspace_lifecycle_and_default_workspace_protection(ws):
    names = [w["name"] for w in client.get("/workspaces").json()]
    assert "My Documents" in names and "Research" in names
    assert client.delete(f"/workspaces/{repo.DEFAULT_WORKSPACE_ID}").status_code == 400
    upload(ws, ("ml_paper.md", PAPER, "x"))
    assert (settings.WORKSPACES_DIR / ws).exists()
    assert client.delete(f"/workspaces/{ws}").status_code == 200
    assert not (settings.WORKSPACES_DIR / ws).exists() and client.get(f"/workspaces/{ws}/documents").status_code == 404
    assert client.get("/workspaces/BAD_ID!/documents").status_code == 422


def test_interrupted_documents_are_recovered_on_startup(ws):
    d = repo.create_document(ws, "half.pdf", "pdf", 10)
    assert repo.recover_interrupted() == 1 and repo.get_document(ws, d["id"])["status"] == "failed"


# ---------------- legacy endpoints + SSRF ----------------
def test_legacy_text_ingest_goes_to_default_workspace():
    r = client.post("/ingest/text", json={"text": "The beta release is scheduled for 14 March this year.", "source_name": "notes"})
    assert r.status_code == 200 and r.json()["chunks_added"] >= 1
    assert client.get("/index/status").json()["documents"] == 1
    client.post("/index/reset")
    assert client.get("/index/status").json()["documents"] == 0


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://localhost/admin", "http://169.254.169.254/latest/meta-data/",
                                 "file:///etc/passwd", "ftp://example.com/x"])
def test_url_ingestion_blocks_internal_and_non_http_targets(url):
    assert client.post("/ingest/url", json={"url": url}).status_code == 400


def test_url_import_reads_public_html_and_revalidates_redirects(monkeypatch):
    import httpx
    from app.rag import web_loader
    monkeypatch.setattr(web_loader.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])

    def handler(req: httpx.Request):
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><head><title>Page</title><script>bad()</script></head><body><p>The budget is 250,000 dollars.</p></body></html>")

    title, text = web_loader.fetch_url_text("https://example.com/p", transport=httpx.MockTransport(handler))
    assert title == "Page" and "250,000" in text and "bad()" not in text

    monkeypatch.setattr(web_loader.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("127.0.0.1" if host == "evil.internal" else "93.184.216.34", 443))])
    redirect = lambda req: httpx.Response(302, headers={"location": "http://evil.internal/secret"})
    with pytest.raises(Exception, match="private"):
        web_loader.fetch_url_text("https://example.com/r", transport=httpx.MockTransport(redirect))
