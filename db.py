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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ug "
                "ON memories(user_id, group_id)"
            )
            conn.commit()

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
            cur = conn.execute(
                "INSERT INTO memories "
                "(user_id, group_id, content, embedding_model, embedding, "
                "created_at, updated_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
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
            return cur.lastrowid  # type: ignore[return-value]

    def get_active_memories(
        self, user_id: str, group_id: str
    ) -> list[tuple[int, str, str, str]]:
        """Returns (id, content, embedding_json, embedding_model) for non-expired rows."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, content, embedding, embedding_model FROM memories "
                "WHERE user_id=? AND group_id=? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (user_id, group_id, now),
            ).fetchall()
        return rows  # type: ignore[return-value]

    def list_memories(
        self, user_id: str, group_id: str
    ) -> list[tuple[int, str, float, float | None]]:
        """Returns (id, content, created_at, expires_at) DESC by created_at."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, content, created_at, expires_at FROM memories "
                "WHERE user_id=? AND group_id=? "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY created_at DESC",
                (user_id, group_id, now),
            ).fetchall()
        return rows  # type: ignore[return-value]

    def delete_memory(self, memory_id: int, user_id: str, group_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE id=? AND user_id=? AND group_id=?",
                (memory_id, user_id, group_id),
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
