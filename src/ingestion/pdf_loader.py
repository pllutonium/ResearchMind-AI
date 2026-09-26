"""
Module: pdf_loader.py
Description: Ingests documents, extracts textual layers, and applies OCR when necessary.
"""
import sys
from typing import List, Dict, Any
import fitz  # PyMuPDF
from src.ingestion.ocr import extract_text_with_ocr


def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts structured page-level text from a specified PDF file.
    Applies OCR fallback for image-based or low-density text pages.

    Args:
        pdf_path (str): File system path to the target PDF document.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing page metadata and text.
    """
    document_pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        extracted_text = page.get_text().strip()

        # Fallback to OCR if textual layer density is insufficient
        if len(extracted_text) < 20:
            extracted_text = extract_text_with_ocr(page)

        document_pages.append({
            "page_number": page_index + 1,
            "text": extracted_text
        })

    doc.close()
    return document_pages


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = load_pdf_text(sys.argv[1])
        print(f"[INFO] Successfully processed {len(result)} pages.")
        if result:
            print("[INFO] Sample extraction from Page 1:")
            print(result[0]["text"][:300])
    else:
        print("[USAGE] python -m src.ingestion.pdf_loader <path_to_pdf>")