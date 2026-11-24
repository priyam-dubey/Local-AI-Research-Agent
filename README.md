# AI Research & Document Intelligence Platform

One local-first application with two experiences that share the same model-provider layer:

| | |
|---|---|
| **AI Research Agent** | A LangGraph team (Planner → Researcher → Writer → Critic) plans, researches (your documents + DuckDuckGo), drafts, self-critiques and exports a PDF report, with live progress. |
| **Document Intelligence Assistant** | Upload PDF, DOCX, TXT, Markdown and CSV files into workspaces, get AI-generated questions from the real text, chat with follow-ups, search one / all documents or **compare** several — every answer grounded in the files and cited by file + page. |

**You choose the model, explicitly:** *Local (Ollama)* keeps prompts and documents on your machine; *Gemini (Online)* uses Google's API with your own key. The app never switches between them silently. Embeddings (a local HuggingFace model) and the FAISS indexes are always local, whichever LLM you pick.

---

## Quick start

Requirements: Python 3.10+, Node 18+ (for the UI), and either [Ollama](https://ollama.com) or a Gemini API key.

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # optional: edit settings

# 2a. Local AI: install Ollama, then
ollama serve &                                           # if it is not already running
ollama pull mistral                                      # or llama3, gemma, ...

# 2b. Online AI: get a key at https://aistudio.google.com/ and either
#     paste it in the app (Model Selection) or set GEMINI_API_KEY in backend/.env

# 3. Frontend
cd ../frontend
npm install
npm run build                                            # the backend will serve it
cd ../backend && uvicorn app.main:app --port 8000        # open http://localhost:8000
```

**Development with hot reload** — two terminals: `cd backend && uvicorn app.main:app --reload` and `cd frontend && npm run dev` → http://localhost:5173 (talks to the API on :8000; override with `VITE_API_URL`).

> The first document upload downloads the embedding model (~90 MB) once. After that, Local mode works offline (except optional web research).

## Using it

1. **Model Selection** (sidebar): *Local Model (Ollama)* or *Gemini (Online)*. For Gemini paste your key → **Validate & save**. The key is sent once to your backend, validated against Google, and held **in backend memory only** — never in the browser.
2. **Document Assistant**: pick/create a workspace, drag in files (several at once). Status goes *Queued → Processing → Processed*. Suggested questions appear (from the real text; *Generate More* adds new, non-duplicate ones). Click one or type your own.
3. **Scope**: click a document to ask only about it, or leave none selected to search all. **Compare Documents** tab: tick 2–5 documents for similarities, differences, conflicting findings, methodologies and conclusions.
4. **New Research**: enter a topic; optionally also use a workspace's documents as local evidence. Watch the agents work, read the report, download the PDF. **Research History** keeps past reports.

## How answers stay grounded (and what that does *not* guarantee)

Defence in depth — the prompt is only one layer:

1. **Retrieval gate before the model is called.** A question needs passages above a similarity threshold *and* its content terms must actually appear in them (or a very strong semantic match). Otherwise the app replies *"I couldn't find this information in the uploaded document."* and **never calls the LLM**.
2. **Constrained prompt.** Excerpts are delimited, untrusted data; the model must answer only from them, say what is missing, and tag each sentence with an excerpt id. Document text can't forge delimiters or become instructions.
3. **Verification after generation.** Sentences with no valid citation, an unknown id, or a number that isn't in the cited excerpt are removed; there is one corrective retry; if nothing verifiable remains the answer is a refusal.
4. **Citations come from retrieved-chunk metadata**, not from model text: file, **page** (PDF), or location (*Section: Methods*, *Rows 26–50*) for formats without real pages. Page numbers are never invented.
5. **Isolation.** One FAISS index per workspace; document scopes are validated against the workspace; comparison retrieves per document and presents each document's excerpts under its own heading.

Answers are shown **only after verification**; the UI streams *stages* (searching → writing → checking), not unverified tokens.

Limits to be aware of: verification checks citations and numbers, not every claim (a model can still misread a cited passage); names and paraphrased facts are not machine-checked; scanned PDFs need OCR (not supported); small local models cite less reliably, so you may see more "removed statements" warnings.

### Calibrating the thresholds
Defaults are starting values for `all-MiniLM-L6-v2`. Measure on *your* documents:
```bash
cd backend
python -m scripts.calibrate_thresholds --docs ./my_docs --questions questions.json
# questions.json: {"answerable": ["..."], "unanswerable": ["..."]}
```
It prints each question's score and suggests settings (false accepts are penalised 3× more than false refusals). The test suite includes a labelled corpus (`backend/tests/fixtures`) showing why *two* signals are used: on it, an unanswerable question out-scored three answerable ones on similarity alone, while term coverage separated them cleanly. Those test thresholds are calibrated for the test embedder, **not** for MiniLM — calibrate before relying on them.

## Configuration (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DEFAULT_PROVIDER` | `ollama` | Used only if a request names no provider. A failing provider is never replaced by the other. |
| `OLLAMA_BASE_URL`, `LLM_MODEL`, `OLLAMA_TIMEOUT` | `http://localhost:11434`, `mistral`, `300` | Local model. |
| `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TIMEOUT` | – , `gemini-2.5-flash`, `60` | Online model (model names change; check Google's list). |
| `EMBEDDING_BACKEND`, `EMBED_MODEL` | `hf`, `sentence-transformers/all-MiniLM-L6-v2` | `hashing` is a fake embedder for tests only. |
| `RETRIEVAL_MIN_SIMILARITY`, `RETRIEVAL_STRONG_SIMILARITY`, `EVIDENCE_MIN_TERM_COVERAGE` | `0.30`, `0.55`, `0.34` | Evidence gate (see above). |
| `QA_TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `6`, `1000`, `200` | Retrieval/chunking. |
| `MAX_UPLOAD_MB`, `MAX_UPLOADS_PER_REQUEST` | `50`, `20` | Upload limits. |
| `MAX_ITERATIONS` | `5` | Critic loop limit for research. |

## API

Interactive docs: `http://localhost:8000/docs`. Errors are `{"error": {"code", "message", "provider"}}`.

| Area | Endpoints |
|---|---|
| Settings / providers | `GET /settings` · `GET /settings/providers/{ollama\|gemini}/health` · `POST /settings/gemini-key` · `DELETE /settings/gemini-key` · `GET /system/status` |
| Workspaces | `GET/POST /workspaces` · `DELETE /workspaces/{id}` |
| Documents | `POST /workspaces/{id}/documents` (multi-file, async processing) · `GET /workspaces/{id}/documents[/{doc}]` · `DELETE /workspaces/{id}/documents/{doc}` |
| Suggested questions | `GET/POST /workspaces/{id}/questions` |
| Chat | `POST /workspaces/{id}/chat` · `POST /workspaces/{id}/chat/stream` (SSE: `stage`, `final`, `error`) · `GET /workspaces/{id}/conversations[/{conv}]` · `DELETE …/{conv}` |
| Research | `POST /research` (`provider`, `model`, `workspace_id`) · `GET /research/{id}/stream` (SSE) · `GET /research/{id}/result` · `GET /research/{id}/report.pdf` · `GET /research` (history) |
| Legacy (deprecated) | `/ingest/{file,url,text}`, `/index/{status,reset}` → operate on the `default` workspace |

Chat body: `{"question", "conversation_id?", "provider?", "model?", "scope": {"mode": "document|all|compare", "document_ids": []}}`.
Answer: `{"answer", "refused", "grounded", "sources": [{"id","filename","page","location","excerpt","score"}], "warnings", "standalone_question", …}`.

## Tests

```bash
cd backend && pip install pytest && python -m pytest -q     # 136 tests, no Ollama / Gemini key / model download needed
cd frontend && npm test                                      # 18 component + flow tests (jsdom)
```
Gemini is always mocked; Ollama is exercised through a mock HTTP transport; the LangGraph workflow, FAISS, PDF/DOCX parsing, the evidence gate, verification and the HTTP API run for real. A ready-made GitHub Actions workflow that runs both suites is in `ci/github-actions.yml` — copy it to `.github/workflows/ci.yml` (GitHub only lets tokens with the `workflow` scope push files under `.github/workflows/`, so it is shipped outside that folder).

## Project layout

```
backend/app/
  providers/   base.py · ollama_provider.py · gemini_provider.py · factory.py   (one interface, explicit selection)
  agents/      planner · researcher · writer · critic                           (provider-agnostic)
  graph.py     LangGraph topology
  rag/         loaders · ingestion(chunking) · embeddings · vector_store(FAISS/workspace) · retrieval(evidence gate)
               document_qa (grounded QA + compare) · question_generator · web_loader (SSRF-guarded)
  services/    repo (SQLite) · document_service · chat_service · research_service · export_service
  api/routes/  settings · documents · chat · research · system · ingest(legacy)
  core/        security (secrets, redaction, safe filenames) · errors
backend/scripts/calibrate_thresholds.py     backend/tests/
frontend/src/  lib (api, sse, citations) · state · components · pages
legacy_streamlit_ui.py                      (old UI, kept until you no longer need it)
```

## Security notes

Implemented: server-side-only Gemini key (memory or `.env`), redaction of keys in errors/logs, file-type + **content** sniffing, size limits, safe unique filenames and temp-file cleanup, per-workspace isolation, SSRF-guarded URL import (public hosts only, redirects re-checked), no pickle loading (indexes are `.faiss` + `.npy` + JSON), no raw HTML rendering in the UI, delimiter-hardened prompts for untrusted documents.

**Not implemented — do not deploy this for multiple users as is:** there is no authentication or per-user data separation, the Gemini key is global to the server process, and workspace ids are not secret. Run it on localhost. Before any shared deployment add authentication, per-user workspaces/secrets, rate limiting and HTTPS. URL import has a small DNS-rebinding window (resolution is checked, then connected separately).

## Troubleshooting

* **"Cannot reach Ollama"** → `ollama serve`; **"model not installed"** → `ollama pull <model>`.
* **Gemini "invalid key"/"rate limit"** → check the key at aistudio.google.com; rate limits are retried with backoff, then reported.
* **Document "Failed: No extractable text"** → scanned PDF (no OCR). **"Re-upload the documents"** → the embedding model changed; the workspace was indexed with another one.
* **Many "Not found" answers** → thresholds too strict for your embedding model/documents; run the calibration script.
* Research uses DuckDuckGo for web evidence; with Local AI your *document* data stays local, but search queries do go to DuckDuckGo.

See `CHANGELOG.md` for what changed from v1.
