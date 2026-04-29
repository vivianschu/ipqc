"""SQLite database schema and CRUD for user accounts and run metadata."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

_DB_PATH = Path(__file__).parent.parent / "data" / "qc.db"


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they do not already exist."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                email         TEXT,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                name          TEXT    NOT NULL,
                upload_date   TEXT    NOT NULL,
                sample_count  INTEGER NOT NULL,
                sample_names  TEXT    NOT NULL,
                summary_json  TEXT,
                data_dir      TEXT    NOT NULL
            );
        """)


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip(), password_hash, datetime.utcnow().isoformat()),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def username_exists(username: str) -> bool:
    return get_user_by_username(username) is not None


def update_password(user_id: int, new_password_hash: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_password_hash, user_id),
        )


# ── Run CRUD ──────────────────────────────────────────────────────────────────

def save_run(
    user_id: int,
    name: str,
    sample_names: list[str],
    summary: dict[str, Any],
    data_dir: str,
) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO runs
               (user_id, name, upload_date, sample_count, sample_names, summary_json, data_dir)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                name,
                datetime.utcnow().isoformat(),
                len(sample_names),
                json.dumps(sample_names),
                json.dumps(summary),
                data_dir,
            ),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]


def get_runs_for_user(user_id: int) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM runs WHERE user_id = ? ORDER BY upload_date DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int, user_id: int) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM runs WHERE id = ? AND user_id = ?",
            (run_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def delete_run(run_id: int, user_id: int) -> None:
    with _conn() as con:
        con.execute(
            "DELETE FROM runs WHERE id = ? AND user_id = ?",
            (run_id, user_id),
        )


def update_run_data_dir(run_id: int, user_id: int, data_dir: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE runs SET data_dir = ? WHERE id = ? AND user_id = ?",
            (data_dir, run_id, user_id),
        )
