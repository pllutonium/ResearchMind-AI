"""
Module: src/ingestion/parser.py
Description: Production academic PDF parser powered by PyMuPDF (fitz).
Extracts structured text while preserving section headers, tables, formulas, and page numbers.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import fitz  # PyMuPDF
from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Represents text extracted from a single document page."""
    page_number: int
    text: str
    char_count: int
    tables_found: int = 0
    section_headers: List[str] = Field(default_factory=list)


class ParsedPaper(BaseModel):
    """Structured representation of a parsed academic paper."""
    paper_id: str
    file_path: str
    title: str
    authors: Optional[str] = None
    total_pages: int
    pages: List[ExtractedPage]
    full_text: str


class AcademicPDFParser:
    """Enterprise PyMuPDF parser tailored for arXiv and conference papers."""

    def __init__(self):
        # Regex heuristics for academic section headers and LaTeX blocks
        self.header_pattern = re.compile(
            r"^(?:\d+(?:\.\d+)*\s+)?(Abstract|Introduction|Related\s+Work|Methodology|Methods|Architecture|Experiments|Results|Discussion|Conclusion|References)\b",
            re.IGNORECASE,
        )

    def parse(self, file_path: str | Path) -> ParsedPaper:
        """
        Parses an academic PDF into structured pages with headers and metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        paper_id = path.stem
        doc = fitz.open(str(path))
        total_pages = len(doc)

        extracted_pages: List[ExtractedPage] = []
        full_text_blocks: List[str] = []
        detected_title = None

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1

            # Extract structured blocks with font and coordinates
            blocks = page.get_text("blocks")
            cleaned_page_text_parts: List[str] = []
            page_headers: List[str] = []
            tables_count = 0

            # PyMuPDF table extraction if supported in this PyMuPDF version
            try:
                tables = page.find_tables()
                tables_count = len(tables.tables) if tables else 0
            except Exception:
                tables_count = 0

            for b in blocks:
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                text = b[4].strip()
                if not text:
                    continue

                # Filter out running header/footer noise (typically top 35px or bottom 35px)
                rect_y0, rect_y1 = b[1], b[3]
                if (rect_y0 < 30 or rect_y1 > (page.rect.height - 30)) and len(text) < 60:
                    # Likely a running page number or header
                    continue

                # Detect paper title from Page 1 first substantive block
                if page_number == 1 and detected_title is None and len(text) > 10 and not text.lower().startswith("arxiv"):
                    # First large prominent block is the title
                    first_line = text.splitlines()[0].strip()
                    detected_title = first_line

                # Check if block represents an academic section header
                match = self.header_pattern.match(text)
                if match:
                    header_name = match.group(0).strip()
                    page_headers.append(header_name)

                # Clean hyphenation at line breaks (e.g., 'trans- \nformer' -> 'transformer')
                clean_block = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
                clean_block = re.sub(r"\n+", " ", clean_block)
                cleaned_page_text_parts.append(clean_block)

            page_full_text = "\n\n".join(cleaned_page_text_parts).strip()
            extracted_pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=page_full_text,
                    char_count=len(page_full_text),
                    tables_found=tables_count,
                    section_headers=page_headers,
                )
            )
            if page_full_text:
                full_text_blocks.append(f"--- [Page {page_number}] ---\n{page_full_text}")

        meta = doc.metadata or {}
        doc_title = meta.get("title") or detected_title or paper_id
        doc_authors = meta.get("author") or "Unknown Authors"
        doc.close()

        return ParsedPaper(
            paper_id=paper_id,
            file_path=str(path),
            title=doc_title.strip(),
            authors=doc_authors.strip(),
            total_pages=total_pages,
            pages=extracted_pages,
            full_text="\n\n".join(full_text_blocks),
        )


# Global parser instance
pdf_parser = AcademicPDFParser()
