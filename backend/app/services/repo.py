"""SQLite persistence: workspaces, documents, suggested questions, conversations, research history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.config import settings
from app.core.errors import NotFoundError

DEFAULT_WORKSPACE_ID = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, file_type TEXT NOT NULL, size_bytes INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL, error TEXT, pages INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_documents_ws ON documents(workspace_id);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, created_at REAL NOT NULL,
    UNIQUE(workspace_id, document_id, text));
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'all', created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY, query TEXT NOT NULL, provider TEXT NOT NULL, model TEXT,
    workspace_id TEXT, status TEXT NOT NULL, report TEXT NOT NULL DEFAULT '',
    plan TEXT NOT NULL DEFAULT '[]', citations TEXT NOT NULL DEFAULT '[]',
    quality REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL, completed_at REAL);
"""


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as c:
        c.executescript(_SCHEMA)
        if not c.execute("SELECT 1 FROM workspaces WHERE id=?", (DEFAULT_WORKSPACE_ID,)).fetchone():
            c.execute("INSERT INTO workspaces VALUES (?,?,?)", (DEFAULT_WORKSPACE_ID, "My Documents", time.time()))


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _row(r: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(r) if r else None


# ---------------- workspaces ----------------
def create_workspace(name: str) -> dict:
    ws = {"id": _new_id(), "name": name.strip()[:80] or "Untitled", "created_at": time.time()}
    with db() as c:
        c.execute("INSERT INTO workspaces VALUES (:id,:name,:created_at)", ws)
    return ws


def list_workspaces() -> list[dict]:
    with db() as c:
        rows = c.execute("""SELECT w.*, (SELECT COUNT(*) FROM documents d WHERE d.workspace_id=w.id) AS document_count
                            FROM workspaces w ORDER BY created_at""").fetchall()
    return [dict(r) for r in rows]


def get_workspace(ws_id: str) -> dict:
    with db() as c:
        row = c.execute("SELECT * FROM workspaces WHERE id=?", (ws_id,)).fetchone()
    if not row:
        raise NotFoundError(f"Workspace '{ws_id}' not found.")
    return dict(row)


def delete_workspace(ws_id: str) -> None:
    with db() as c:
        c.execute("DELETE FROM workspaces WHERE id=?", (ws_id,))


# ---------------- documents ----------------
def create_document(ws_id: str, filename: str, file_type: str, size: int) -> dict:
    doc = {"id": _new_id(), "workspace_id": ws_id, "filename": filename, "file_type": file_type,
           "size_bytes": size, "status": "queued", "error": None, "pages": 0, "chunks": 0,
           "created_at": time.time()}
    with db() as c:
        c.execute("""INSERT INTO documents VALUES
            (:id,:workspace_id,:filename,:file_type,:size_bytes,:status,:error,:pages,:chunks,:created_at)""", doc)
    return doc


def update_document(doc_id: str, **fields: Any) -> None:
    allowed = {"status", "error", "pages", "chunks", "size_bytes"}
    cols = [k for k in fields if k in allowed]
    if not cols:
        return
    with db() as c:
        c.execute(f"UPDATE documents SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
                  [fields[k] for k in cols] + [doc_id])


def list_documents(ws_id: str, status: Optional[str] = None) -> list[dict]:
    q, args = "SELECT * FROM documents WHERE workspace_id=?", [ws_id]
    if status:
        q, args = q + " AND status=?", args + [status]
    with db() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC", args).fetchall()]


def get_document(ws_id: str, doc_id: str) -> dict:
    """Scoped by workspace: a document id from another workspace is 'not found'."""
    with db() as c:
        row = c.execute("SELECT * FROM documents WHERE id=? AND workspace_id=?", (doc_id, ws_id)).fetchone()
    if not row:
        raise NotFoundError(f"Document '{doc_id}' not found in this workspace.")
    return dict(row)


def delete_document_row(doc_id: str) -> None:
    with db() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        c.execute("DELETE FROM questions WHERE document_id=?", (doc_id,))


def recover_interrupted() -> int:
    """Documents left queued/processing by a crash or restart can never finish."""
    with db() as c:
        cur = c.execute("UPDATE documents SET status='failed', error='Processing was interrupted by a restart. Please upload again.' "
                        "WHERE status IN ('queued','processing')")
        return cur.rowcount


# ---------------- suggested questions ----------------
def add_questions(ws_id: str, doc_id: str, texts: list[str]) -> None:
    with db() as c:
        for t in texts:
            c.execute("INSERT OR IGNORE INTO questions (workspace_id, document_id, text, created_at) VALUES (?,?,?,?)",
                      (ws_id, doc_id or "", t, time.time()))


def list_questions(ws_id: str, doc_id: str = "") -> list[str]:
    with db() as c:
        rows = c.execute("SELECT text FROM questions WHERE workspace_id=? AND document_id=? ORDER BY id",
                         (ws_id, doc_id or "")).fetchall()
    return [r["text"] for r in rows]


# ---------------- conversations ----------------
def create_conversation(ws_id: str, title: str, mode: str) -> dict:
    now = time.time()
    conv = {"id": _new_id(), "workspace_id": ws_id, "title": title.strip()[:80] or "New chat",
            "mode": mode, "created_at": now, "updated_at": now}
    with db() as c:
        c.execute("INSERT INTO conversations VALUES (:id,:workspace_id,:title,:mode,:created_at,:updated_at)", conv)
    return conv


def get_conversation(ws_id: str, conv_id: str) -> dict:
    with db() as c:
        row = c.execute("SELECT * FROM conversations WHERE id=? AND workspace_id=?", (conv_id, ws_id)).fetchone()
    if not row:
        raise NotFoundError("Conversation not found in this workspace.")
    return dict(row)


def list_conversations(ws_id: str) -> list[dict]:
    with db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM conversations WHERE workspace_id=? ORDER BY updated_at DESC", (ws_id,)).fetchall()]


def delete_conversation(ws_id: str, conv_id: str) -> None:
    get_conversation(ws_id, conv_id)
    with db() as c:
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def add_message(conv_id: str, role: str, content: str, meta: Optional[dict] = None) -> dict:
    msg = {"id": _new_id(), "conversation_id": conv_id, "role": role, "content": content,
           "meta": json.dumps(meta or {}), "created_at": time.time()}
    with db() as c:
        c.execute("INSERT INTO messages VALUES (:id,:conversation_id,:role,:content,:meta,:created_at)", msg)
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (msg["created_at"], conv_id))
    return {**msg, "meta": meta or {}}


def list_messages(conv_id: str, limit: Optional[int] = None) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, rowid", (conv_id,)).fetchall()
    out = [{**dict(r), "meta": json.loads(r["meta"] or "{}")} for r in rows]
    return out[-limit:] if limit else out


# ---------------- research history ----------------
def save_research(s: dict) -> None:
    with db() as c:
        c.execute("""INSERT OR REPLACE INTO research_sessions
            (id, query, provider, model, workspace_id, status, report, plan, citations, quality, created_at, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (s["id"], s["query"], s["provider"], s.get("model"), s.get("workspace_id"), s["status"],
                   s.get("report", ""), json.dumps(s.get("plan", [])), json.dumps(s.get("citations", [])),
                   s.get("quality", 0.0), s["created_at"], time.time()))


def list_research(limit: int = 50) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT id, query, provider, model, status, quality, created_at, completed_at "
                         "FROM research_sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_research(session_id: str) -> Optional[dict]:
    with db() as c:
        row = c.execute("SELECT * FROM research_sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["plan"], d["citations"] = json.loads(d["plan"]), json.loads(d["citations"])
    return d
