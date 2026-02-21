import sqlite3
from typing import List, Dict
from contextlib import contextmanager
from app.config import settings, logger


class ChatMemory:
    """Quản lý lịch sử chat với SQLite."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self._init_db()
        logger.info(f"Chat memory initialized: {self.db_path}")

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB Error: {e}")
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    chunks_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON conversations(created_at)")

    def save_conversation(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: List[str],
        chunks_count: int,
    ) -> int:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,),
            )
            conn.execute(
                "UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            cursor = conn.execute(
                """INSERT INTO conversations
                   (session_id, query, answer, sources, chunks_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, query, answer, ",".join(sources), chunks_count),
            )
            return cursor.lastrowid

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT query, answer, created_at
                   FROM conversations
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_recent_history_for_context(self, session_id: str, last_n: int = 3) -> str:
        """Trả về chuỗi lịch sử ngắn để đưa vào LLM context."""
        history = self.get_session_history(session_id, limit=last_n)
        if not history:
            return ""
        lines = []
        for h in history:
            lines.append(f"User: {h['query']}")
            # Giới hạn độ dài answer để không làm quá dài context
            answer_preview = h["answer"][:500]
            lines.append(f"Assistant: {answer_preview}{'...' if len(h['answer']) > 500 else ''}")
        return "\n".join(lines)

    def delete_session(self, session_id: str) -> bool:
        with self.get_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            logger.info(f"Deleted session {session_id}")
            return True

    def get_stats(self) -> Dict:
        with self.get_connection() as conn:
            total_convos = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return {
                "total_conversations": total_convos,
                "total_sessions": total_sessions,
                "db_path": self.db_path,
            }


# Singleton instance
chat_memory = ChatMemory()
