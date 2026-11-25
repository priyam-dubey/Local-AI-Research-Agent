# Changelog

## v2.0.0-dev — Phases 3–6 (document intelligence, React UI, docs)

### Added
- **Workspaces & documents**: PDF / DOCX / TXT / Markdown / CSV upload (multi-file, async processing with status),
  page-aware extraction, chunking with filename/page/location metadata, local embeddings, **one FAISS index per workspace**,
  deletion with index rebuild, SQLite registry (documents, questions, conversations, research history).
- **Grounded document QA** with a retrieval evidence gate, constrained prompt, post-generation verification
  (citations + numbers), metadata-built citations, follow-up question rewriting, prompt-injection hardening.
- **AI-generated suggested questions** from the document text (deduplicated, answerability-filtered, "generate more").
- **Multi-document** scopes: one document, all documents, comparison (per-document context).
- **React + TypeScript + Tailwind UI** (dashboard, document assistant with citations/streaming stages, compare, research
  with live agent feed, history, settings) — served by FastAPI after `npm run build`.
- Research agent can use a workspace's documents; irrelevant local results now correctly fall back to the web.
- Embedding model warm-up at startup with UI status; `scripts/prefetch_models.py`, `scripts/migrate_v1.py`.
- `scripts/calibrate_thresholds.py`, CI workflow template (`ci/github-actions.yml`), 136 backend + 18 frontend tests.

### Changed
- Vector store: LangChain/pickle FAISS → direct FAISS (cosine/IP) per workspace, no pickle files.
- `/ingest/*` and `/index/*` are deprecated aliases for the `default` workspace; `/ingest/url` is now SSRF-guarded.
- Web search now uses the maintained `ddgs` package (`duckduckgo-search` is unmaintained/renamed) with a fallback to the legacy one.
- `langchain-huggingface` replaced by `sentence-transformers` directly; `python-docx` added.
- Existing v1 FAISS indexes are **not** migrated; re-upload documents.

## v2.0.0-dev — Phase 1 & 2 (repo restructure + provider abstraction)

### Added
- `backend/app/providers/`: `BaseLLM`, `OllamaProvider` (HTTP), `GeminiProvider` (`google-genai`),
  factory. Typed errors (missing/invalid key, rate limit, timeout, unavailable, model not found).
- Explicit provider selection per research run (`provider`, `model` in `POST /research`).
  No silent fallback between providers.
- `POST/DELETE /settings/gemini-key` (validated, held in backend memory only),
  `GET /settings`, `GET /settings/providers/{provider}/health`.
- `GET /research/{id}/report.pdf` (PDF export moved to the backend).
- Retries with backoff for Gemini 429/5xx; timeouts for both providers.
- Test suite (59 tests) that needs no Ollama, Gemini key or embedding model.

### Fixed
- SSE delivered earlier messages again at every graph step (duplicate events).
- Critic could approve after step 1 of N and silently skip the rest of the plan.
- `final_report` stayed empty when the plan finished before the critic approved.
- Critic reported a fake `quality_score=1.0` at the iteration limit.
- Writer truncated the *tail* of the evidence (later plan steps were dropped).
- A draft report was generated after every research step (now once per pass).
- Upload path traversal (`../` in filename), no size limit, whole-file-in-memory read.
- PDF export could fail on `&`/`<` in headings and bullets.
- `.gitignore` mangled and ignored `.env.example`; stray committed `mnt/` directory; unused `unstructured` dependency.

### Changed (breaking)
- Code now lives in the `backend/app` package (`from app.config import settings`).
  Run with `cd backend && uvicorn app.main:app --reload`.
- Graph: research all plan steps → draft → critic; critic GAPS become extra plan steps.
- `langchain-ollama` removed (Ollama is called directly).
- CORS now also allows the Vite dev server (5173).
- Legacy `ui.py` moved to `legacy_streamlit_ui.py`; it still works against the same endpoints.

### Known / deferred
- Relevance threshold + web-fallback fix for the research agent: needs the workspace/retrieval work (Phase 3) and calibration with the real embedding model.
- Single global FAISS index, in-memory research sessions: replaced in Phase 3.
- `/ingest/url` has no SSRF protection yet (Phase 3).
- No authentication: the server-held Gemini key is for single-user local use only.
