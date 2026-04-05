"""
src/db/sqlite_db.py — SQLite Database Layer
=============================================
WHY SQLite for user data?
- Chat history, user preferences = structured, relational data.
- SQLite is a file-based SQL database — zero setup, zero server.
- Perfect for personal/session data that doesn't need scale yet.
- When you need to scale, swap to PostgreSQL with minimal changes.

WHAT IS STORED HERE?
  Table: chat_messages
    session_id  TEXT    → groups messages into a conversation
    role        TEXT    → "user" or "assistant"
    content     TEXT    → the actual message text
    created_at  TEXT    → ISO timestamp for ordering

HOW SQLITE WORKS IN PYTHON:
  - sqlite3 is in Python's standard library (no install needed).
  - We use context managers (with conn:) so connections always close cleanly.
  - WAL mode = Write-Ahead Logging → allows concurrent reads while writing.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


class SQLiteDB:
    """
    Thin wrapper around SQLite.
    All SQL is in this class — nothing else in the app touches SQL directly.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """
        Creates tables if they don't exist yet.
        Called once on startup.
        """
        with self._connect() as conn:
            conn.executescript("""
                -- Enable WAL mode for better concurrent read performance
                PRAGMA journal_mode=WAL;

                -- chat_messages: stores every message exchanged in a session
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                    content     TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                -- Index so fetching a session's history is fast
                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_messages(session_id, created_at);
            """)
        logger.info("SQLite schema ready at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """
        Opens a connection to the SQLite file.
        Using `with self._connect() as conn` auto-commits on success,
        auto-rolls back on exception, and closes the connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # Rows behave like dicts: row["column"]
        return conn

    def execute(self, sql: str, params: tuple = ()):
        """
        Utility: run any SQL. Used by the health check.
        """
        with self._connect() as conn:
            conn.execute(sql, params)

    # ── Chat History ────────────────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str):
        """
        Saves one message (user or assistant) to the chat history.
        """
        sql = """
            INSERT INTO chat_messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """
        # We use UTC timestamps so timezone issues never corrupt ordering.
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(sql, (session_id, role, content, now))

    def get_recent_history(self, session_id: str, limit: int = 6) -> List[dict]:
        """
        Returns the last `limit` message pairs for a session.
        Used to give the LLM conversation context ("memory").

        Returns a list like:
          [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        sql = """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (session_id, limit * 2)).fetchall()

        # Rows are newest-first from DESC order — reverse to get oldest-first
        # (LLMs expect history oldest → newest)
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


# ── Factory function ──────────────────────────────────────────────────────────
def init_sqlite(db_path: str) -> SQLiteDB:
    """
    Called once in app.py's create_app().
    Returns a ready-to-use SQLiteDB instance.
    """
    import os
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return SQLiteDB(db_path)
