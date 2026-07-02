"""M6.5 human-in-the-loop: a durable review queue for escalated questions.

When the gate escalates a turn, the question lands here. An operator lists open items, claims and
answers one, and the closed row (which holds the answer) is the source of truth the flywheel
(M7.3) re-indexes as gold; a verified JSONL is written as a convenience cache alongside it. This
SQLite store is the durable part (it survives restarts). The LangGraph checkpointer persists a
graph run's state within a process (MemorySaver); cross-restart resume needs SqliteSaver, wired
at M9. SQLite now, Postgres at M9 (same schema), mirroring the auth store.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
import uuid

_KEYS = ["id", "domain", "message_id", "question", "route", "status", "answer", "answered_by",
         "created_at", "claimed_at", "resolved_at"]
_STALE_CLAIM_SECONDS = 900.0  # a claim older than this is abandoned and can be taken over


class ReviewQueue:
    def __init__(self, path: str, verified_path: str | None = None) -> None:
        self.path = path
        self.verified_path = verified_path
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS review_queue ("
                "id TEXT PRIMARY KEY, domain TEXT, message_id TEXT, question TEXT NOT NULL, "
                "route TEXT, status TEXT NOT NULL DEFAULT 'open', answer TEXT, answered_by TEXT, "
                "created_at REAL, claimed_at REAL, resolved_at REAL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rq_status ON review_queue(status)")
            conn.commit()

    def _conn(self):
        # WAL + a busy timeout so concurrent admin writes wait briefly instead of raising
        # "database is locked"
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return contextlib.closing(conn)

    def enqueue(self, question: str, *, domain: str | None = None, message_id: str | None = None,
                route: str | None = None, now: float | None = None) -> str:
        """Add an escalated question. A retry of the same turn (same message_id) returns the
        existing open item instead of enqueuing a duplicate."""
        with self._conn() as conn:
            if message_id:
                existing = conn.execute(
                    "SELECT id FROM review_queue WHERE message_id = ? AND status = 'open'",
                    (message_id,)).fetchone()
                if existing:
                    return existing[0]
            item_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO review_queue (id, domain, message_id, question, route, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, domain, message_id, question, route,
                 now if now is not None else time.time()))
            conn.commit()
        return item_id

    def list_open(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, domain, question, route, created_at FROM review_queue "
                "WHERE status = 'open' ORDER BY created_at LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "domain": r[1], "question": r[2], "route": r[3], "created_at": r[4]}
                for r in rows]

    def claim(self, item_id: str, operator: str, now: float | None = None) -> bool:
        """Lock an item to one operator (open -> claimed). An abandoned claim (older than the
        stale window) can be taken over. Returns False if it is fresh-claimed by someone else or
        already closed, so two operators cannot both hold the same item."""
        stamp = now if now is not None else time.time()
        cutoff = stamp - _STALE_CLAIM_SECONDS
        with self._conn() as conn:
            row = conn.execute(
                "UPDATE review_queue SET status = 'claimed', answered_by = ?, claimed_at = ? "
                "WHERE id = ? AND (status = 'open' OR (status = 'claimed' AND claimed_at < ?)) "
                "RETURNING id", (operator, stamp, item_id, cutoff)).fetchone()
            conn.commit()
        return row is not None

    def list_actionable(self, operator: str, limit: int = 50, now: float | None = None) -> list:
        """What an operator can act on: open items to claim, their own claimed items to answer,
        and stale claims they can take over. Fixes the flow where a claimed item would vanish."""
        cutoff = (now if now is not None else time.time()) - _STALE_CLAIM_SECONDS
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, domain, question, route, created_at, status, answered_by "
                "FROM review_queue WHERE status = 'open' "
                "OR (status = 'claimed' AND answered_by = ?) "
                "OR (status = 'claimed' AND claimed_at < ?) "
                "ORDER BY created_at LIMIT ?", (operator, cutoff, limit)).fetchall()
        return [{"id": r[0], "domain": r[1], "question": r[2], "route": r[3], "created_at": r[4],
                 "status": r[5], "claimed_by": r[6] if r[5] == "claimed" else None} for r in rows]

    def get(self, item_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT {} FROM review_queue WHERE id = ?".format(", ".join(_KEYS)),
                (item_id,)).fetchone()
        return dict(zip(_KEYS, row)) if row else None

    def resolve(self, item_id: str, answer: str, answered_by: str,
                now: float | None = None) -> bool:
        """Close an open item with a human answer. Returns False if it was already closed or is
        missing, so two operators cannot both answer the same item. The closed row is the source
        of truth (it holds the answer); the verified JSONL is a best-effort cache."""
        stamp = now if now is not None else time.time()
        with self._conn() as conn:
            # RETURNING gives the question/domain from the same atomic UPDATE, so there is no race
            # between closing the row and reading it back on a second connection. An open item can
            # be answered directly; a claimed one only by the operator who claimed it (row lock).
            row = conn.execute(
                "UPDATE review_queue SET status = 'closed', answer = ?, answered_by = ?, "
                "resolved_at = ? WHERE id = ? AND "
                "(status = 'open' OR (status = 'claimed' AND answered_by = ?)) "
                "RETURNING question, domain",
                (answer, answered_by, stamp, item_id, answered_by)).fetchone()
            conn.commit()
        if row is None:
            return False
        if self.verified_path:
            self._append_verified({
                "question": row[0], "answer": answer, "domain": row[1],
                "answered_by": answered_by, "ts": stamp, "source": "hitl"})
        return True

    def closed_since(self, ts: float = 0.0) -> list[dict]:
        """Resolved items for the flywheel to re-index, straight from the durable rows so it does
        not depend on the verified JSONL cache."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT question, answer, domain, answered_by, resolved_at FROM review_queue "
                "WHERE status = 'closed' AND resolved_at >= ? ORDER BY resolved_at",
                (ts,)).fetchall()
        return [{"question": r[0], "answer": r[1], "domain": r[2], "answered_by": r[3],
                 "resolved_at": r[4]} for r in rows]

    def _append_verified(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.verified_path) or ".", exist_ok=True)
        with open(self.verified_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
