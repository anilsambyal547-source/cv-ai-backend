"""
Email tracking and scheduling using SQLite.
Stores sent emails and scheduled emails to prevent duplicate outreach and enable future sending.
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class EmailTracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sent_emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    candidate_email TEXT,
                    candidate_name TEXT,
                    subject TEXT,
                    template_variant TEXT,
                    sent_at TIMESTAMP,
                    message_id TEXT,
                    thread_id TEXT,
                    status TEXT DEFAULT 'sent'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    candidate_email TEXT,
                    candidate_name TEXT,
                    subject TEXT,
                    template_variant TEXT,
                    scheduled_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.commit()

    def log_sent(
        self,
        candidate_id: str,
        candidate_email: str,
        candidate_name: str,
        subject: str,
        template_variant: str,
        message_id: str,
        thread_id: Optional[str] = None,
    ):
        """Record a sent email."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sent_emails
                (candidate_id, candidate_email, candidate_name, subject, template_variant, sent_at, message_id, thread_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    candidate_email,
                    candidate_name,
                    subject,
                    template_variant,
                    datetime.now(),
                    message_id,
                    thread_id,
                ),
            )
            conn.commit()

    def was_emailed_recently(
        self,
        candidate_email: str,
        cooldown_days: int = 30,
    ) -> bool:
        """Check if candidate was emailed within cooldown period."""
        cutoff = datetime.now() - timedelta(days=cooldown_days)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM sent_emails WHERE candidate_email = ? AND sent_at > ?",
                (candidate_email, cutoff),
            )
            count = cur.fetchone()[0]
            return count > 0

    def schedule_email(
        self,
        candidate_id: str,
        candidate_email: str,
        candidate_name: str,
        subject: str,
        template_variant: str,
        scheduled_time: datetime,
    ):
        """Store an email to be sent later."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scheduled_emails
                (candidate_id, candidate_email, candidate_name, subject, template_variant, scheduled_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    candidate_email,
                    candidate_name,
                    subject,
                    template_variant,
                    scheduled_time,
                ),
            )
            conn.commit()

    def get_pending_scheduled(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve scheduled emails that are due for sending."""
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT id, candidate_id, candidate_email, candidate_name, subject, template_variant
                FROM scheduled_emails
                WHERE status = 'pending' AND scheduled_time <= ?
                ORDER BY scheduled_time
                LIMIT ?
                """,
                (now, limit),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def mark_scheduled_sent(self, scheduled_id: int, message_id: str, thread_id: Optional[str] = None):
        """Mark a scheduled email as sent and move to sent_emails."""
        with sqlite3.connect(self.db_path) as conn:
            # First get the scheduled record
            cur = conn.execute(
                "SELECT * FROM scheduled_emails WHERE id = ?",
                (scheduled_id,),
            )
            row = cur.fetchone()
            if row:
                # Insert into sent_emails
                conn.execute(
                    """
                    INSERT INTO sent_emails
                    (candidate_id, candidate_email, candidate_name, subject, template_variant, sent_at, message_id, thread_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row[1], row[2], row[3], row[4], row[5],
                        datetime.now(), message_id, thread_id,
                    ),
                )
                # Delete from scheduled
                conn.execute("DELETE FROM scheduled_emails WHERE id = ?", (scheduled_id,))
            conn.commit()

    def mark_scheduled_failed(self, scheduled_id: int):
        """Mark a scheduled email as failed (optional)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE scheduled_emails SET status = 'failed' WHERE id = ?",
                (scheduled_id,),
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Return basic stats for reporting."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM sent_emails")
            sent_count = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM scheduled_emails WHERE status='pending'")
            scheduled_count = cur.fetchone()[0]
            return {"sent": sent_count, "scheduled": scheduled_count}