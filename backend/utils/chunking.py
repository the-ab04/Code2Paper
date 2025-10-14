# backend/utils/chunking.py

from typing import List, Dict
from .text_cleaning import clean_text, split_into_sentences

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks for embedding/indexing.
    
    Args:
        text (str): Cleaned input text.
        chunk_size (int): Max characters per chunk.
        overlap (int): Characters to overlap between chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # move window with overlap

    return chunks


def chunk_by_sentences(
    text: str,
    max_sentences: int = 5,
    overlap: int = 1
) -> List[str]:
    """
    Split text into chunks based on sentences instead of raw characters.
    Useful for academic papers where sentence boundaries matter.

    Args:
        text (str): Input text.
        max_sentences (int): Max sentences per chunk.
        overlap (int): Overlap in number of sentences.

    Returns:
        List[str]: List of sentence-based chunks.
    """
    sentences = split_into_sentences(clean_text(text))
    if not sentences:
        return []

    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max_sentences - overlap

    return chunks


def prepare_chunks_with_metadata(
    text: str,
    paper_id: str,
    section: str,
    method: str = "char"
) -> List[Dict]:
    """
    Create chunks along with metadata for Qdrant or FAISS.
    
    Args:
        text (str): Raw section text.
        paper_id (str): Unique identifier (DOI, arXiv ID, etc.).
        section (str): Section name (e.g., Introduction, Methods).
        method (str): "char" or "sentence".
    
    Returns:
        List[Dict]: [{"text": ..., "metadata": {...}}, ...]
    """
    if method == "sentence":
        chunks = chunk_by_sentences(text)
    else:
        chunks = chunk_text(text)

    return [
        {
            "text": c,
            "metadata": {
                "paper_id": paper_id,
                "section": section
            }
        }
        for c in chunks
    ]
