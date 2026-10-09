"""
Module: src/ingestion/docstore.py
Description: Production Key-Value document store for Parent Chunks in hierarchical RAG.
Persists parent documents in an atomic SQLite database with an in-memory cache.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import sqlite3
import threading
from pydantic import BaseModel, Field
from config.settings import settings


class ParentDocument(BaseModel):
    """Data model for a parent chunk stored in the docstore."""
    parent_id: str
    paper_id: str
    page_number: int
    title: str = "Unknown Paper"
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentStore:
    """Thread-safe on-disk key-value store for Parent Documents backed by SQLite."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.DOCSTORE_PATH
        self._cache: Dict[str, ParentDocument] = {}
        self._lock = threading.Lock()
        self._init_db()
        self.load()

    def _get_connection(self) -> sqlite3.Connection:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.storage_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes SQLite schema for parent documents."""
        with self._lock:
            # Check for legacy JSON migration
            if self.storage_path.exists() and not self.storage_path.suffix.startswith(".sqlite"):
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            self._cache[k] = ParentDocument(**v)
                except Exception:
                    pass

            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS parent_docs (
                            parent_id TEXT PRIMARY KEY,
                            paper_id TEXT,
                            page_number INTEGER,
                            title TEXT,
                            text TEXT,
                            metadata TEXT
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_paper ON parent_docs(paper_id)")
            finally:
                conn.close()

    def load(self) -> None:
        """Loads parent documents from SQLite into the in-memory cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT parent_id, paper_id, page_number, title, text, metadata FROM parent_docs")
                for row in cursor.fetchall():
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    self._cache[row["parent_id"]] = ParentDocument(
                        parent_id=row["parent_id"],
                        paper_id=row["paper_id"],
                        page_number=row["page_number"],
                        title=row["title"] or "Unknown Paper",
                        text=row["text"],
                        metadata=meta,
                    )
            except Exception as e:
                print(f"[WARN] Failed to load docstore from SQLite ({self.storage_path}): {e}")
            finally:
                conn.close()

    def save(self) -> None:
        """Flushes in-memory cache to SQLite in a single transaction (idempotent)."""
        with self._lock:
            if not self._cache:
                return
            conn = self._get_connection()
            try:
                with conn:
                    rows = [
                        (
                            doc.parent_id,
                            doc.paper_id,
                            doc.page_number,
                            doc.title,
                            doc.text,
                            json.dumps(doc.metadata, ensure_ascii=False),
                        )
                        for doc in self._cache.values()
                    ]
                    conn.executemany("""
                        INSERT OR REPLACE INTO parent_docs
                        (parent_id, paper_id, page_number, title, text, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
            finally:
                conn.close()

    def put(self, parent_doc: ParentDocument) -> None:
        """Stores a single parent document atomically in SQLite and memory."""
        with self._lock:
            self._cache[parent_doc.parent_id] = parent_doc
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO parent_docs
                        (parent_id, paper_id, page_number, title, text, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        parent_doc.parent_id,
                        parent_doc.paper_id,
                        parent_doc.page_number,
                        parent_doc.title,
                        parent_doc.text,
                        json.dumps(parent_doc.metadata, ensure_ascii=False),
                    ))
            finally:
                conn.close()

    def put_many(self, docs: List[ParentDocument]) -> None:
        """Stores a batch of parent documents atomically in a single SQLite transaction."""
        if not docs:
            return
        with self._lock:
            for doc in docs:
                self._cache[doc.parent_id] = doc

            conn = self._get_connection()
            try:
                with conn:
                    rows = [
                        (
                            doc.parent_id,
                            doc.paper_id,
                            doc.page_number,
                            doc.title,
                            doc.text,
                            json.dumps(doc.metadata, ensure_ascii=False),
                        )
                        for doc in docs
                    ]
                    conn.executemany("""
                        INSERT OR REPLACE INTO parent_docs
                        (parent_id, paper_id, page_number, title, text, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
            finally:
                conn.close()

    def get(self, parent_id: str) -> Optional[ParentDocument]:
        """Retrieves a parent document by its unique ID."""
        with self._lock:
            if parent_id in self._cache:
                return self._cache[parent_id]

            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT parent_id, paper_id, page_number, title, text, metadata FROM parent_docs WHERE parent_id = ?",
                    (parent_id,),
                )
                row = cursor.fetchone()
                if row:
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    doc = ParentDocument(
                        parent_id=row["parent_id"],
                        paper_id=row["paper_id"],
                        page_number=row["page_number"],
                        title=row["title"] or "Unknown Paper",
                        text=row["text"],
                        metadata=meta,
                    )
                    self._cache[parent_id] = doc
                    return doc
                return None
            finally:
                conn.close()

    def get_many(self, parent_ids: List[str]) -> List[ParentDocument]:
        """Retrieves multiple parent documents, preserving request order."""
        docs = []
        for pid in parent_ids:
            doc = self.get(pid)
            if doc is not None:
                docs.append(doc)
        return docs

    def all_ids(self) -> List[str]:
        """Returns all registered parent IDs."""
        with self._lock:
            return list(self._cache.keys())

    def count(self) -> int:
        """Returns total number of parent documents stored."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clears all records in SQLite and memory."""
        with self._lock:
            self._cache.clear()
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM parent_docs")
            finally:
                conn.close()


# Global document store singleton
DocumentStoreType = DocumentStore
DocStore = DocumentStore
docstore = DocumentStore()
