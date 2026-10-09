"""
Module: metadata_store.py
Description: Lightweight SQLite repository for paper registry and corpus tracking.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import METADATA_DB_PATH


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(METADATA_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the metadata schema if not already present."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_paper_metadata(
    paper_id: str,
    filename: str,
    page_count: int,
    chunk_count: int,
    title: Optional[str] = None,
) -> None:
    """Inserts or updates paper indexing records."""
    init_db()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, filename, title, page_count, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                filename=excluded.filename,
                title=excluded.title,
                page_count=excluded.page_count,
                chunk_count=excluded.chunk_count,
                created_at=excluded.created_at
            """,
            (paper_id, filename, title or paper_id, page_count, chunk_count, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_paper_metadata(paper_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata record for a specific paper."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_all_papers() -> List[Dict[str, Any]]:
    """Returns all registered paper records ordered by recency."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute("SELECT * FROM papers ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
