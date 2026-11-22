# Changelog

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
