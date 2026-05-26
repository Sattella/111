from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class MemoryDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    mem_no        INTEGER NOT NULL DEFAULT 0,
                    user_id       TEXT    NOT NULL,
                    group_id      TEXT    NOT NULL DEFAULT '',
                    content       TEXT    NOT NULL,
                    embedding_model TEXT  NOT NULL,
                    embedding     TEXT    NOT NULL,
                    created_at    REAL    NOT NULL,
                    updated_at    REAL    NOT NULL,
                    expires_at    REAL
                )
            """)
            # Migration: add mem_no to existing databases
            try:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN mem_no INTEGER NOT NULL DEFAULT 0"
                )
                # Backfill sequential mem_no per user+group ordered by id
                rows = conn.execute(
                    "SELECT id, user_id, group_id FROM memories ORDER BY user_id, group_id, id"
                ).fetchall()
                counters: dict[tuple, int] = {}
                for row_id, uid, gid in rows:
                    key = (uid, gid)
                    counters[key] = counters.get(key, 0) + 1
                    conn.execute(
                        "UPDATE memories SET mem_no=? WHERE id=?",
                        (counters[key], row_id),
                    )
            except Exception:
                pass  # Column already exists
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ug "
                "ON memories(user_id, group_id)"
            )
            conn.commit()

    def _next_mem_no(self, conn: sqlite3.Connection, user_id: str, group_id: str) -> int:
        """Return the smallest positive integer not currently used as mem_no for user+group."""
        rows = conn.execute(
            "SELECT mem_no FROM memories WHERE user_id=? AND group_id=? ORDER BY mem_no",
            (user_id, group_id),
        ).fetchall()
        used = {r[0] for r in rows}
        n = 1
        while n in used:
            n += 1
        return n

    def add_memory(
        self,
        user_id: str,
        group_id: str,
        content: str,
        embedding_model: str,
        embedding: list[float],
        expires_at: float | None = None,
    ) -> int:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            mem_no = self._next_mem_no(conn, user_id, group_id)
            conn.execute(
                "INSERT INTO memories "
                "(mem_no, user_id, group_id, content, embedding_model, embedding, "
                "created_at, updated_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mem_no,
                    user_id,
                    group_id,
                    content,
                    embedding_model,
                    json.dumps(embedding),
                    now,
                    now,
                    expires_at,
                ),
            )
            conn.commit()
            return mem_no

    def get_active_memories(
        self, user_id: str, group_id: str
    ) -> list[tuple[int, str, str, str]]:
        """Returns (mem_no, content, embedding_json, embedding_model) for non-expired rows."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT mem_no, content, embedding, embedding_model FROM memories "
                "WHERE user_id=? AND group_id=? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (user_id, group_id, now),
            ).fetchall()
        return rows  # type: ignore[return-value]

    def list_memories(
        self, user_id: str, group_id: str
    ) -> list[tuple[int, str, float, float | None]]:
        """Returns (mem_no, content, created_at, expires_at) DESC by created_at."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT mem_no, content, created_at, expires_at FROM memories "
                "WHERE user_id=? AND group_id=? "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY created_at DESC",
                (user_id, group_id, now),
            ).fetchall()
        return rows  # type: ignore[return-value]

    def delete_memory(self, mem_no: int, user_id: str, group_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE mem_no=? AND user_id=? AND group_id=?",
                (mem_no, user_id, group_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def clear_memories(self, user_id: str, group_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE user_id=? AND group_id=?",
                (user_id, group_id),
            )
            conn.commit()
            return cur.rowcount

    def cleanup_expired(self) -> int:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM memories "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            conn.commit()
            return cur.rowcount
