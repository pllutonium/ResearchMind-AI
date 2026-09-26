"""
Module: chroma_store.py
Description: Manages vector indexing, persistent storage, and nearest-neighbor search via ChromaDB.
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Suppress telemetry flags
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


@contextmanager
def suppress_stderr():
    """Temporarily suppresses stderr to silence internal third-party telemetry warnings."""
    original_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = original_stderr


with suppress_stderr():
    import chromadb
    from chromadb.api.models.Collection import Collection

from config import CHROMA_DB_DIR, VECTOR_DB_COLLECTION_NAME

_client_instance: Optional[chromadb.PersistentClient] = None
_collection_instance: Optional[Collection] = None


def _get_collection() -> Collection:
    """
    Retrieves or initializes the persistent ChromaDB collection.

    Returns:
        Collection: Active ChromaDB collection reference.
    """
    global _client_instance, _collection_instance
    if _client_instance is None:
        with suppress_stderr():
            _client_instance = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            _collection_instance = _client_instance.get_or_create_collection(name=VECTOR_DB_COLLECTION_NAME)
    return _collection_instance


def add_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """
    Persists document chunks, embeddings, and associated metadata to the vector database.

    Args:
        chunks (List[Dict[str, Any]]): List of serialized document chunks.
        embeddings (List[List[float]]): Corresponding vector representations.
    """
    collection = _get_collection()
    with suppress_stderr():
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"paper_id": c["paper_id"], "page_number": c["page_number"]}
                for c in chunks
            ],
        )


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    paper_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes an approximate nearest neighbor search against stored document vectors.

    Args:
        query_embedding (List[float]): Encoded query vector.
        top_k (int): Number of closest matches to retrieve. Defaults to 5.
        paper_id (Optional[str]): If given, restrict the search to chunks belonging
            to this single paper only (needed so agents don't mix content from
            different papers together).

    Returns:
        List[Dict[str, Any]]: Ranked list of matching chunks with similarity distance scores.
    """
    collection = _get_collection()
    where_filter = {"paper_id": paper_id} if paper_id else None
    with suppress_stderr():
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

    matched_chunks: List[Dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        matched_chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "paper_id": results["metadatas"][0][i]["paper_id"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
        })
    return matched_chunks


def count_stored_chunks() -> int:
    """
    Returns the total count of vectorized segments currently stored in the database.

    Returns:
        int: Number of records in the active collection.
    """
    with suppress_stderr():
        return _get_collection().count()


def list_indexed_paper_ids() -> List[str]:
    """
    Returns the distinct list of paper_id values currently indexed, so the UI
    (or an agent) can let the person pick which paper to work with instead of
    a hardcoded default.
    """
    with suppress_stderr():
        collection = _get_collection()
        everything = collection.get(include=["metadatas"])
    paper_ids = {m["paper_id"] for m in everything.get("metadatas", []) if m}
    return sorted(paper_ids)


class VectorStore:
    """
    Thin object-oriented wrapper around the module-level functions above.

    research_agents.py and dashboard/app.py both expect a *class* they can
    instantiate (`VectorStore()`), so this wraps the existing free functions
    rather than duplicating the ChromaDB logic. It also takes plain query
    *text* (not a pre-computed embedding) so callers don't need to import
    the embedder themselves.
    """

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Imported lazily to avoid a circular import at module load time
        # (embedder.py doesn't import this module, but keeping it local
        # here keeps chroma_store.py usable on its own too).
        from src.embeddings.embedder import embed_single_text

        query_vector = embed_single_text(query_text)
        return search_similar_chunks(query_vector, top_k=n_results, paper_id=paper_id)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        add_chunks(chunks, embeddings)

    def count(self) -> int:
        return count_stored_chunks()

    def list_paper_ids(self) -> List[str]:
        return list_indexed_paper_ids()