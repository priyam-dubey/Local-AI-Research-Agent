import json

import numpy as np
import pytest

from app.config import settings
from app.core.errors import EmbeddingError, IngestionError, UnsupportedTypeError
from app.rag import embeddings
from app.rag.embeddings import HashingEmbedder, get_embedder
from app.rag.ingestion import chunk_sections
from app.rag.loaders import Section, detect_type, load_document
from app.rag.vector_store import clear_registry, drop_index, get_index
from tests.helpers import make_docx, make_pdf


def write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ------------------------------------------------------------------ loaders
def test_pdf_keeps_real_page_numbers(tmp_path):
    p = write(tmp_path, "a.pdf", make_pdf(["First page about apples.", "Second page about pears.", "Third page about plums."]))
    secs = load_document(p, "pdf")
    assert [s.page for s in secs] == [1, 2, 3] and "pears" in secs[1].text and secs[1].location == "Page 2"


def test_docx_headings_and_tables(tmp_path):
    data = make_docx([("h", "Methods"), ("p", "We used a survey."), ("h", "Results"), ("p", "Response rate was high.")],
                     table=[["Metric", "Value"], ["Respondents", "412"]])
    secs = load_document(write(tmp_path, "r.docx", data), "docx")
    locs = [s.location for s in secs]
    assert "Section: Methods" in locs and "Section: Results" in locs
    assert any("Respondents | 412" in s.text for s in secs)
    assert all(s.page is None for s in secs)             # DOCX has no reliable pages: never invent them


def test_text_encodings(tmp_path):
    cp = "Café résumé naïve — the dataset is small.".encode("cp1252")
    assert "Café" in load_document(write(tmp_path, "c.txt", cp), "txt")[0].text
    u16 = "Unicode text in UTF-16".encode("utf-16")
    assert "Unicode" in load_document(write(tmp_path, "u.txt", u16), "txt")[0].text


def test_markdown_sections(tmp_path):
    md = b"# Title\nintro\n\n## Methods\nwe did x\n\n## Results\nwe got y\n"
    locs = [s.location for s in load_document(write(tmp_path, "n.md", md), "md")]
    assert "Section: Methods" in locs and "Section: Results" in locs


def test_csv_rows_become_labelled_chunks(tmp_path):
    rows = "name,score\n" + "\n".join(f"user{i},{i}" for i in range(60))
    secs = load_document(write(tmp_path, "d.csv", rows.encode()), "csv")
    assert secs[0].location == "Rows 1-25" and "name: user0; score: 0" in secs[0].text and secs[-1].location == "Rows 51-60"


def test_scanned_pdf_without_text_is_reported(tmp_path):
    with pytest.raises(IngestionError, match="scanned"):
        load_document(write(tmp_path, "s.pdf", make_pdf([" ", " "])), "pdf")


@pytest.mark.parametrize("name,data,ftype,exc", [
    ("fake.pdf", b"this is not a pdf", "pdf", UnsupportedTypeError),
    ("fake.docx", b"plain text pretending", "docx", UnsupportedTypeError),
    ("bin.txt", b"\x00\x01\x02binary\x00", "txt", UnsupportedTypeError),
    ("empty.txt", b"", "txt", IngestionError),
])
def test_content_sniffing_rejects_disguised_files(tmp_path, name, data, ftype, exc):
    with pytest.raises(exc):
        load_document(write(tmp_path, name, data), ftype)


def test_corrupted_docx_zip(tmp_path):
    with pytest.raises(IngestionError):
        load_document(write(tmp_path, "bad.docx", b"PK\x03\x04garbagegarbage"), "docx")


def test_unsupported_extension():
    with pytest.raises(UnsupportedTypeError):
        detect_type("malware.exe")


# ------------------------------------------------------------------ chunking
def test_chunks_carry_metadata_and_never_cross_pages():
    secs = [Section("alpha " * 400, "Page 1", 1), Section("beta " * 400, "Page 2", 2)]
    chunks = chunk_sections(secs, "doc1", "paper.pdf")
    assert len(chunks) > 2
    for c in chunks:
        assert c["doc_id"] == "doc1" and c["filename"] == "paper.pdf" and c["chunk_id"].startswith("doc1-")
        assert ("alpha" in c["text"]) != ("beta" in c["text"])             # one page per chunk
        assert c["page"] == (1 if "alpha" in c["text"] else 2)
    assert len(chunks[0]["text"]) <= settings.CHUNK_SIZE + 50


# ------------------------------------------------------------------ embeddings
def test_embeddings_are_normalised_deterministic_and_independent_of_llm():
    e = get_embedder()
    v = e.embed_documents(["MNIST handwritten digits", "MNIST handwritten digits", "something else entirely"])
    assert v.shape == (3, e.dim) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.allclose(v[0], v[1]) and float(v[0] @ v[2]) < 0.5
    assert isinstance(e, HashingEmbedder)


# ------------------------------------------------------------------ FAISS store
def _chunks(doc, texts):
    return [{"chunk_id": f"{doc}-{i}", "doc_id": doc, "filename": f"{doc}.txt", "page": None, "location": "", "ordinal": i, "text": t}
            for i, t in enumerate(texts)]


def test_search_ranks_filters_and_persists():
    e = get_embedder()
    idx = get_index("ws1")
    a = _chunks("A", ["cats purr and sleep all day", "the stock market fell sharply"])
    b = _chunks("B", ["dogs bark loudly at night"])
    idx.add(a, e.embed_documents([c["text"] for c in a]), e.name)
    idx.add(b, e.embed_documents([c["text"] for c in b]), e.name)

    top = idx.search(e.embed_query("cats sleeping"), 2)
    assert top[0][0]["doc_id"] == "A" and "cats" in top[0][0]["text"]
    only_b = idx.search(e.embed_query("cats sleeping"), 5, doc_ids={"B"})
    assert {c["doc_id"] for c, _ in only_b} == {"B"}

    clear_registry()                                                       # simulate restart
    reloaded = get_index("ws1")
    assert len(reloaded.chunks) == 3 and reloaded.search(e.embed_query("dogs barking"), 1)[0][0]["doc_id"] == "B"
    assert (settings.WORKSPACES_DIR / "ws1" / "index.faiss").exists()
    assert not list((settings.WORKSPACES_DIR / "ws1").glob("*.pkl"))        # no pickle files


def test_delete_document_rebuilds_index():
    e = get_embedder()
    idx = get_index("ws1")
    for d, t in (("A", ["apples are red fruit"]), ("B", ["bananas are yellow fruit"])):
        ch = _chunks(d, t)
        idx.add(ch, e.embed_documents(t), e.name)
    assert idx.remove_document("A") == 1
    assert idx.doc_ids() == {"B"} and idx.index.ntotal == 1
    assert all(c["doc_id"] == "B" for c, _ in idx.search(e.embed_query("apples"), 3))


def test_workspaces_are_isolated():
    e = get_embedder()
    ch = _chunks("A", ["secret merger plan with Acme"])
    get_index("ws1").add(ch, e.embed_documents([c["text"] for c in ch]), e.name)
    assert get_index("ws2").search(e.embed_query("secret merger plan"), 5) == []
    drop_index("ws1")
    assert not (settings.WORKSPACES_DIR / "ws1").exists()


def test_embedding_model_mismatch_is_refused_not_silently_mixed():
    e = get_embedder()
    idx = get_index("ws1")
    ch = _chunks("A", ["hello world"])
    idx.add(ch, e.embed_documents(["hello world"]), e.name)
    with pytest.raises(EmbeddingError, match="Re-upload"):
        idx.add(ch, e.embed_documents(["hello world"]), "some-other-model")
