"""
Module: ocr.py
Description: Handles optical character recognition (OCR) fallback for rasterized PDF pages.
"""
import io
import os
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from config import TESSERACT_CMD

logger = logging.getLogger(__name__)

# Configure Tesseract binary path if discovered
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text_with_ocr(page: fitz.Page, dpi: int = 200) -> str:
    """
    Renders a PDF page to a rasterized image and extracts textual content via Tesseract OCR.
    Gracefully handles environments where Tesseract is not installed.

    Args:
        page (fitz.Page): The PyMuPDF page object to process.
        dpi (int): Resolution density for page rasterization. Defaults to 200.

    Returns:
        str: Extracted raw text content, or empty string on OCR failure.
    """
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        image_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(image, lang="eng")
        return extracted_text.strip()
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "[OCR WARNING] Tesseract OCR is not installed or not found on system PATH. "
            "Skipping OCR extraction for image-based page."
        )
        return ""
    except Exception as e:
        logger.warning(f"[OCR WARNING] OCR extraction failed: {e}")
        return ""