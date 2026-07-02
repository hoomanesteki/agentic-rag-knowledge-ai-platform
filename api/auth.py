"""Auth: SQLite-backed users, bcrypt password hashing, JWT sessions, Turnstile verification.

SQLite runs anywhere with no infra; the same schema runs on Postgres at deploy (swap the
connection, M9.3). Tokens travel in the Authorization header, so there are no cookies and no
CSRF surface. One connection per call keeps it thread-safe under the server's threadpool.
"""
from __future__ import annotations

import contextlib
import sqlite3
import time

import bcrypt
import httpx
import jwt

_ALGO = "HS256"
_TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


# Compared against for unknown users so login timing does not reveal whether a user exists.
DUMMY_HASH = hash_password("timing-equalizer")


def create_access_token(username: str, role: str, secret: str, expires_min: int = 60) -> str:
    now = int(time.time())
    payload = {"sub": username, "role": role, "iat": now, "exp": now + expires_min * 60}
    return jwt.encode(payload, secret, algorithm=_ALGO)


def decode_token(token: str, secret: str) -> dict:
    # raises jwt.PyJWTError on a bad signature, expiry, or a missing required claim
    return jwt.decode(token, secret, algorithms=[_ALGO], options={"require": ["exp", "sub"]})


def verify_turnstile(token: str | None, secret: str) -> bool:
    if not secret:
        return True  # dev bypass when Turnstile is not configured
    if not token:
        return False
    try:
        resp = httpx.post(_TURNSTILE_URL, data={"secret": secret, "response": token}, timeout=10)
        return bool(resp.json().get("success"))
    except Exception:
        return False


class UserStore:
    def __init__(self, path: str) -> None:
        self.path = path
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, "
                "role TEXT NOT NULL DEFAULT 'customer')")
            conn.commit()

    def _conn(self):
        # closing() so the connection is released, not left to GC
        return contextlib.closing(sqlite3.connect(self.path))

    def get(self, username: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT username, password_hash, role FROM users WHERE username = ?",
                (username,)).fetchone()
        return {"username": row[0], "password_hash": row[1], "role": row[2]} if row else None

    def create(self, username: str, password_hash: str, role: str = "customer") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role))
            conn.commit()


def seed_demo_user(store: UserStore, username: str, password: str,
                   role: str = "customer") -> None:
    if not store.get(username):
        store.create(username, hash_password(password), role)
