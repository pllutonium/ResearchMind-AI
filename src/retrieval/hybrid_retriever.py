"""
Module: src/retrieval/hybrid_retriever.py
Description: Production Hybrid Retriever combining Sparse BM25 and Dense ChromaDB vectors.
Applies Reciprocal Rank Fusion (RRF, k=60), maps child hits to Parent Documents, and passes candidates to Cross-Encoder.
"""
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os
import json
import re
import warnings

warnings.filterwarnings("ignore")

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.ingestion.docstore import ParentDocument, docstore
from src.ingestion.chunking import ChildChunk
from src.retrieval.reranker import reranker


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 matching."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """Enterprise Hybrid Retriever combining BM25, ChromaDB dense search, and Cross-Encoder reranking."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
        collection_name: str = "researchmind_agentic_chunks",
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.bm25_path = bm25_path or settings.BM25_INDEX_PATH
        self.collection_name = collection_name

        # 1. Initialize Persistent ChromaDB
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 2. Embedding Model Singleton
        self._embedder = None

        # 3. BM25 Index & Corpus Registry (Idempotent child_id mapping)
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_store: Dict[str, Dict[str, Any]] = {}
        self.bm25_doc_ids: List[str] = []
        self.bm25_parent_map: Dict[str, str] = {}
        self.bm25_corpus: List[List[str]] = []
        self._load_bm25()

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._embedder

    def _rebuild_bm25_structures(self) -> None:
        """Reconstructs BM25 lookup tables and BM25Okapi model from idempotent store."""
        self.bm25_doc_ids = list(self._bm25_store.keys())
        self.bm25_parent_map = {cid: self._bm25_store[cid]["parent_id"] for cid in self.bm25_doc_ids}
        self.bm25_corpus = [self._bm25_store[cid]["tokens"] for cid in self.bm25_doc_ids]
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25 = None

    def _load_bm25(self) -> None:
        """Loads cached BM25 index from disk if present."""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "r", encoding="utf-8") as f:
                    self._bm25_store = json.load(f)
                self._rebuild_bm25_structures()
            except Exception as e:
                # Attempt legacy pickle recovery if file was previously pickled
                try:
                    import pickle
                    with open(self.bm25_path, "rb") as f:
                        data = pickle.load(f)
                        doc_ids = data.get("doc_ids", [])
                        parent_map = data.get("parent_map", {})
                        corpus = data.get("corpus", [])
                        for cid, tokens in zip(doc_ids, corpus):
                            self._bm25_store[cid] = {
                                "parent_id": parent_map.get(cid, ""),
                                "tokens": tokens,
                            }
                        self._rebuild_bm25_structures()
                        self._save_bm25()  # Migrate to secure JSON
                except Exception as e2:
                    print(f"[WARN] Failed to load BM25 index: {e} / {e2}. Index will be rebuilt upon ingestion.")
                    self.bm25 = None

    def _save_bm25(self) -> None:
        """Saves current BM25 index and doc maps to disk safely as JSON."""
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.bm25_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._bm25_store, f)
        if os.path.exists(temp_path):
            os.replace(temp_path, self.bm25_path)

    def index_chunks(self, child_chunks: List[ChildChunk]) -> int:
        """
        Indexes child chunks into both ChromaDB (dense) and BM25 (sparse).
        Ensures strict idempotence to prevent state drift and duplication.
        Returns total count of chunks indexed in this batch.
        """
        if not child_chunks:
            return 0

        # A. Index into ChromaDB
        embedder = self._get_embedder()
        texts = [c.text for c in child_chunks]
        ids = [c.child_id for c in child_chunks]
        metadatas = [
            {
                "parent_id": c.parent_id,
                "paper_id": c.paper_id,
                "page_number": c.page_number,
                "title": c.title,
                "child_index": c.metadata.get("child_index", 0),
            }
            for c in child_chunks
        ]

        # Batch embedding generation
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        # Chroma upsert in batches of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        # B. Index into BM25 idempotently
        for c in child_chunks:
            tokens = _tokenize(c.text)
            self._bm25_store[c.child_id] = {
                "parent_id": c.parent_id,
                "tokens": tokens,
            }

        self._rebuild_bm25_structures()
        self._save_bm25()

        return len(child_chunks)

    def search_dense(self, query: str, top_k: int = 15) -> List[Tuple[str, str, float]]:
        """
        Dense nearest-neighbor query in ChromaDB.
        Returns list of (child_id, parent_id, score).
        """
        if self.collection.count() == 0:
            return []

        embedder = self._get_embedder()
        query_vector = embedder.encode([query], show_progress_bar=False).tolist()
        
        limit = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=limit,
            include=["metadatas", "distances"],
        )

        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            for child_id, meta, dist in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            ):
                parent_id = meta.get("parent_id", "")
                # Cosine distance to similarity score
                similarity = 1.0 - max(0.0, float(dist))
                dense_hits.append((child_id, parent_id, similarity))
        return dense_hits

    def search_sparse(self, query: str, top_k: int = 15) -> List[Tuple[str, str, float]]:
        """
        Sparse BM25 query over chunk tokens.
        Returns list of (child_id, parent_id, score).
        """
        if self.bm25 is None or not self.bm25_doc_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        sparse_hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                child_id = self.bm25_doc_ids[idx]
                parent_id = self.bm25_parent_map.get(child_id, "")
                sparse_hits.append((child_id, parent_id, float(scores[idx])))
        return sparse_hits

    def retrieve_candidates_rrf(
        self,
        query: str,
        pool_size: int = settings.INITIAL_CANDIDATE_POOL,
        k: int = settings.RRF_K,
    ) -> List[ParentDocument]:
        """
        Combines Dense and Sparse hits using Reciprocal Rank Fusion (RRF).
        Aggregates RRF scores directly onto Parent Documents.
        """
        dense_hits = self.search_dense(query, top_k=pool_size)
        sparse_hits = self.search_sparse(query, top_k=pool_size)

        parent_rrf_scores: Dict[str, float] = {}

        # 1. Accumulate RRF for Dense ranks
        for rank, (_child_id, parent_id, _score) in enumerate(dense_hits):
            if parent_id:
                parent_rrf_scores[parent_id] = parent_rrf_scores.get(parent_id, 0.0) + (1.0 / (k + rank + 1))

        # 2. Accumulate RRF for Sparse ranks
        for rank, (_child_id, parent_id, _score) in enumerate(sparse_hits):
            if parent_id:
                parent_rrf_scores[parent_id] = parent_rrf_scores.get(parent_id, 0.0) + (1.0 / (k + rank + 1))

        # Sort parents by combined RRF score descending
        ranked_parent_ids = sorted(
            parent_rrf_scores.keys(),
            key=lambda pid: parent_rrf_scores[pid],
            reverse=True,
        )[:pool_size]

        # Fetch actual parent documents from docstore
        candidate_docs = docstore.get_many(ranked_parent_ids)
        return candidate_docs

    def retrieve(
        self,
        query: str,
        final_top_k: int = settings.FINAL_TOP_K,
        candidate_pool: int = settings.INITIAL_CANDIDATE_POOL,
    ) -> List[ParentDocument]:
        """
        Complete end-to-end retrieval:
        1. Hybrid Dense + Sparse RRF to get top candidates.
        2. Cross-Encoder re-ranking over candidate parent documents.
        3. Return top-4 reranked parent documents.
        """
        candidates = self.retrieve_candidates_rrf(query, pool_size=candidate_pool)
        if not candidates:
            return []

        # Re-rank candidate parent documents with Cross-Encoder
        reranked_tuples = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_top_k,
        )

        return [doc for doc, _score in reranked_tuples]


# Global hybrid retriever singleton
hybrid_retriever = HybridRetriever()
