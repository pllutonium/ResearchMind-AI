"""
Module: src/ingestion/chunking.py
Description: Parent-Document (hierarchical) chunking engine for academic RAG.
Produces large parent context windows for synthesis and small child chunks for high-precision vector search.
Features continuous document streaming across page boundaries to prevent arbitrary truncation.
"""
from typing import List, Tuple, Dict, Any, NamedTuple
import uuid
import re
from pydantic import BaseModel, Field
from config.settings import settings
from src.ingestion.parser import ParsedPaper, ExtractedPage
from src.ingestion.docstore import ParentDocument, docstore


class ChildChunk(BaseModel):
    """Fine-grained child chunk indexed into ChromaDB."""
    child_id: str
    parent_id: str
    paper_id: str
    page_number: int
    title: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class _StreamToken(NamedTuple):
    word: str
    page_number: int
    section_headers: List[str]


def estimate_tokens(text: str) -> int:
    """Estimates subword token counts using regex tokenization for realistic LLM budgets."""
    if not text:
        return 0
    # Counts word tokens and punctuation symbols individually to match BPE/WordPiece behavior
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


class HierarchicalChunker:
    """Splits academic papers into synchronized Parent-Child chunk hierarchies."""

    def __init__(
        self,
        parent_size: int = settings.PARENT_CHUNK_SIZE,
        parent_overlap: int = settings.PARENT_CHUNK_OVERLAP,
        child_size: int = settings.CHILD_CHUNK_SIZE,
        child_overlap: int = settings.CHILD_CHUNK_OVERLAP,
    ):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def _split_into_windows(self, words: List[str], chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
        """
        Splits a sequence of words into overlapping windows.
        Returns list of tuples: (chunk_text, token_count).
        """
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            window = words[i : i + chunk_size]
            if not window:
                continue
            text = " ".join(window).strip()
            if text:
                chunks.append((text, estimate_tokens(text)))
            if i + chunk_size >= len(words):
                break
        return chunks

    def chunk_paper(self, paper: ParsedPaper) -> Tuple[List[ParentDocument], List[ChildChunk]]:
        """
        Processes a parsed paper as a continuous token stream across page boundaries,
        creating linked Parent and Child chunks without page-boundary truncation.
        """
        parent_docs: List[ParentDocument] = []
        child_chunks: List[ChildChunk] = []

        # 1. Build a continuous token stream from all non-empty pages
        stream: List[_StreamToken] = []
        for page in paper.pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            words = page_text.split()
            for w in words:
                stream.append(_StreamToken(word=w, page_number=page.page_number, section_headers=page.section_headers))

        if not stream:
            return [], []

        # 2. Window across continuous stream for Parent chunks
        step = max(1, self.parent_size - self.parent_overlap)
        parent_index = 0

        for i in range(0, len(stream), step):
            window = stream[i : i + self.parent_size]
            if not window:
                continue

            window_words = [t.word for t in window]
            p_text = " ".join(window_words).strip()
            if not p_text:
                continue

            primary_page = window[0].page_number
            end_page = window[-1].page_number
            p_tokens = estimate_tokens(p_text)

            # Deduplicate headers in order of appearance
            headers_seen = set()
            headers = []
            for t in window:
                for h in t.section_headers:
                    if h not in headers_seen:
                        headers_seen.add(h)
                        headers.append(h)

            parent_id = f"{paper.paper_id}_p{primary_page}_{parent_index}_{uuid.uuid4().hex[:6]}"
            parent_doc = ParentDocument(
                parent_id=parent_id,
                paper_id=paper.paper_id,
                page_number=primary_page,
                title=paper.title,
                text=p_text,
                metadata={
                    "token_count": p_tokens,
                    "authors": paper.authors or "",
                    "section_headers": headers,
                    "page_end": end_page,
                    "is_multipage": primary_page != end_page,
                },
            )
            parent_docs.append(parent_doc)

            # 3. Generate Child Chunks strictly inside this Parent Document
            child_windows = self._split_into_windows(
                words=window_words,
                chunk_size=self.child_size,
                overlap=self.child_overlap,
            )

            for c_idx, (c_text, c_tokens) in enumerate(child_windows):
                child_id = f"{parent_id}_c{c_idx}"
                child_chunk = ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    paper_id=paper.paper_id,
                    page_number=primary_page,
                    title=paper.title,
                    text=c_text,
                    metadata={
                        "parent_id": parent_id,
                        "paper_id": paper.paper_id,
                        "page_number": primary_page,
                        "page_end": end_page,
                        "title": paper.title,
                        "child_index": c_idx,
                        "token_count": c_tokens,
                    },
                )
                child_chunks.append(child_chunk)

            parent_index += 1
            if i + self.parent_size >= len(stream):
                break

        # Automatically store parent documents in persistent docstore
        docstore.put_many(parent_docs)

        return parent_docs, child_chunks


# Global hierarchical chunker singleton
hierarchical_chunker = HierarchicalChunker()
