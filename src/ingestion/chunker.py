"""
Module: chunker.py
Description: Splits extracted text streams into structured chunks with contextual overlap.
"""
from typing import List, Dict, Any, Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP

DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]


def _get_overlap_prefix(chunk: str, overlap: int) -> str:
    """Extracts a tail slice from the chunk that snaps to a word boundary."""
    if overlap <= 0 or not chunk:
        return ""
    tail = chunk[-overlap:]
    first_space = tail.find(" ")
    if first_space != -1 and first_space < len(tail) - 1:
        return tail[first_space + 1 :].strip()
    return tail.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits text into chunks respecting semantic boundaries (paragraphs, sentences, words)
    rather than slicing mid-word or mid-sentence.

    Args:
        text (str): Input textual data.
        chunk_size (int): Maximum length per segment in characters.
        overlap (int): Desired contextual overlap between consecutive segments.
        separators (Optional[List[str]]): Ordered list of separators to split on.

    Returns:
        List[str]: Array of extracted, coherent text chunks.
    """
    if not text or not text.strip():
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS

    def _split_recursive(t: str, current_seps: List[str]) -> List[str]:
        t = t.strip()
        if not t:
            return []
        if len(t) <= chunk_size:
            return [t]

        # Find the highest-priority separator present in text
        chosen_sep = ""
        for s in current_seps:
            if s in t:
                chosen_sep = s
                break

        if not chosen_sep:
            # Fallback character split if no further separator applies
            step = max(1, chunk_size - overlap)
            fallback_chunks = [t[i : i + chunk_size].strip() for i in range(0, len(t), step)]
            return [c for c in fallback_chunks if c]

        parts = t.split(chosen_sep)
        next_seps = current_seps[current_seps.index(chosen_sep) + 1 :]
        result: List[str] = []
        current_chunk = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If part is still larger than chunk_size, recursively split it with finer separators
            if len(part) > chunk_size and next_seps:
                sub_chunks = _split_recursive(part, next_seps)
                for sc in sub_chunks:
                    if current_chunk:
                        if len(current_chunk) + len(chosen_sep) + len(sc) <= chunk_size:
                            current_chunk += chosen_sep + sc
                        else:
                            result.append(current_chunk)
                            ol = _get_overlap_prefix(current_chunk, overlap)
                            current_chunk = (ol + " " + sc).strip() if (ol and len(ol) + len(sc) + 1 <= chunk_size) else sc
                    else:
                        current_chunk = sc
                continue

            if not current_chunk:
                current_chunk = part
            elif len(current_chunk) + len(chosen_sep) + len(part) <= chunk_size:
                current_chunk += chosen_sep + part
            else:
                result.append(current_chunk)
                ol = _get_overlap_prefix(current_chunk, overlap)
                current_chunk = (ol + " " + part).strip() if (ol and len(ol) + len(part) + 1 <= chunk_size) else part

        if current_chunk:
            result.append(current_chunk)

        return result

    return _split_recursive(text, seps)



def chunk_pages(pages: List[Dict[str, Any]], paper_id: str) -> List[Dict[str, Any]]:
    """
    Transforms page-level data structures into indexed chunk objects with metadata.

    Args:
        pages (List[Dict[str, Any]]): Extracted document pages.
        paper_id (str): Unique document identifier.

    Returns:
        List[Dict[str, Any]]: Serialized chunk objects ready for embedding generation.
    """
    indexed_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page in pages:
        page_segments = chunk_text(page["text"])
        for segment in page_segments:
            indexed_chunks.append({
                "chunk_id": f"{paper_id}_chunk_{chunk_counter}",
                "paper_id": paper_id,
                "page_number": page["page_number"],
                "text": segment,
            })
            chunk_counter += 1

    return indexed_chunks