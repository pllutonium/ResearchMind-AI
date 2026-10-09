"""
Module: src/retrieval/reranker.py
Description: Production Cross-Encoder re-ranking engine using ms-marco-MiniLM-L-6-v2.
Scores (query, parent_document) pairs to surface the top-4 most semantically relevant documents.
"""
from typing import List, Tuple, Optional
import os
import warnings

# Suppress warnings during cross-encoder initialization
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.settings import settings
from src.ingestion.docstore import ParentDocument


class CrossEncoderReranker:
    """Re-ranks candidate parent documents using a high-precision Cross-Encoder."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy-loads CrossEncoder singleton to conserve memory until query time."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"[WARN] Failed to load CrossEncoder ({self.model_name}): {e}. Fallback to score preservation.")
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[ParentDocument],
        top_k: int = settings.FINAL_TOP_K,
    ) -> List[Tuple[ParentDocument, float]]:
        """
        Re-ranks a list of candidate parent documents against the user query.
        Returns top_k (ParentDocument, score) tuples sorted by descending relevance.
        """
        if not documents:
            return []

        # Remove duplicate parent documents if any
        seen_ids = set()
        unique_docs: List[ParentDocument] = []
        for doc in documents:
            if doc.parent_id not in seen_ids:
                seen_ids.add(doc.parent_id)
                unique_docs.append(doc)

        model = self._get_model()
        if model is None:
            # Fallback: preserve existing order with mock decaying scores
            return [(doc, 1.0 / (idx + 1)) for idx, doc in enumerate(unique_docs[:top_k])]

        try:
            pairs = [[query, doc.text] for doc in unique_docs]
            scores = model.predict(pairs)
            
            # Combine documents with predicted relevance scores
            doc_scores = list(zip(unique_docs, [float(s) for s in scores]))
            # Sort descending by cross-encoder score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:top_k]
        except Exception as e:
            print(f"[WARN] Cross-Encoder prediction failed: {e}. Returning unranked candidates.")
            return [(doc, 0.5) for doc in unique_docs[:top_k]]


# Global cross-encoder reranker singleton
reranker = CrossEncoderReranker()
