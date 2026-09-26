"""
Script: ingest.py
Description: Pipeline orchestrator for document ingestion, processing, embedding, and vector storage.
"""
from pathlib import Path
from typing import List
from tqdm import tqdm

from config import RAW_PDFS_DIR
from src.ingestion.pipeline import ingest_document
from src.vectorstore.chroma_store import count_stored_chunks


def process_single_pdf(pdf_path: Path) -> int:
    """
    Executes the ingestion lifecycle for a single document via the unified pipeline.

    Args:
        pdf_path (Path): Path to the target PDF document.

    Returns:
        int: Number of chunks extracted and indexed.
    """
    print(f"\n[INFO] Processing document: {pdf_path.name}")
    summary = ingest_document(pdf_path)

    if summary["status"] != "success":
        print(f"       [WARNING] Could not index document. Status: {summary['status']}")
        return 0

    print(f"       Extracted {summary['page_count']} page(s), indexed {summary['chunk_count']} chunk(s).")
    return summary["chunk_count"]



def main() -> None:
    """
    Identifies all raw PDF assets and coordinates pipeline execution.
    """
    pdf_files: List[Path] = list(RAW_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF documents located in directory: {RAW_PDFS_DIR}")
        print("        Please provide at least one PDF file before running ingestion.")
        return

    print(f"[INFO] Discovered {len(pdf_files)} document(s) for ingestion.")

    total_chunks = 0
    for pdf_path in tqdm(pdf_files, desc="Ingestion Progress"):
        total_chunks += process_single_pdf(pdf_path)

    print("\n" + "=" * 50)
    print(f"[COMPLETED] Successfully ingested {total_chunks} chunk(s) during this session.")
    print(f"[DATABASE] Total vectorized records in storage: {count_stored_chunks()}")
    print("=" * 50)


if __name__ == "__main__":
    main()