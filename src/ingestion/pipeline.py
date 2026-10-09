"""
Module: pipeline.py
Description: Unified ingestion pipeline for loading, chunking, embedding, and indexing documents.
"""
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

from src.ingestion.pdf_loader import load_pdf_text
from src.ingestion.chunker import chunk_pages
from src.embeddings.embedder import embed_texts
from src.vectorstore.chroma_store import add_chunks
from src.db.metadata_store import save_paper_metadata

logger = logging.getLogger(__name__)


def ingest_document(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Executes the end-to-end ingestion lifecycle for a single PDF document:
    1. Extracts structured page text (with OCR fallback for scanned pages).
    2. Generates boundary-aware semantic chunks.
    3. Computes dense embedding vectors.
    4. Persists chunks and embeddings to ChromaDB.
    5. Stores metadata registry record in SQLite.

    Args:
        pdf_path (Union[str, Path]): File system path to the target PDF document.

    Returns:
        Dict[str, Any]: Ingestion summary containing paper_id, chunks count, and page count.
    """
    path_obj = Path(pdf_path)
    paper_id = path_obj.stem

    # Step 1: Text extraction
    pages = load_pdf_text(str(path_obj))
    if not pages:
        return {"paper_id": paper_id, "chunk_count": 0, "page_count": 0, "status": "empty"}

    # Extract paper title candidate from the first non-empty lines of page 1 if available
    first_page_text = pages[0].get("text", "").strip()
    title_candidate = first_page_text.split("\n")[0][:150] if first_page_text else paper_id

    # Step 2: Semantic boundary-aware chunking
    chunks = chunk_pages(pages, paper_id=paper_id)
    if not chunks:
        return {
            "paper_id": paper_id,
            "chunk_count": 0,
            "page_count": len(pages),
            "status": "no_text_extracted",
        }

    # Step 3: Vector embeddings
    texts_only = [c["text"] for c in chunks]
    embeddings = embed_texts(texts_only)

    # Step 4: Storage ingestion
    add_chunks(chunks, embeddings)

    # Step 5: Persistent catalog registry
    save_paper_metadata(
        paper_id=paper_id,
        filename=path_obj.name,
        page_count=len(pages),
        chunk_count=len(chunks),
        title=title_candidate,
    )

    return {
        "paper_id": paper_id,
        "filename": path_obj.name,
        "title": title_candidate,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "status": "success",
    }
