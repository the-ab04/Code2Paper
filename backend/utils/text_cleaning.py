# backend/utils/text_cleaning.py

import re
import unicodedata
from typing import List

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces/newlines into single ones.
    """
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces/newlines
    return text.strip()


def remove_non_ascii(text: str) -> str:
    """
    Remove or replace non-ASCII characters (like smart quotes, dashes).
    Keeps useful symbols by normalizing to closest ASCII.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(text: str) -> str:
    """
    Apply common cleaning steps to raw text.
    - Normalize whitespace
    - Replace special quotes/dashes
    - Strip leading/trailing junk
    """
    if not text:
        return ""

    # Standardize quotes and dashes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter based on punctuation.
    Useful for chunking academic text into smaller pieces.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def strip_references_section(text: str) -> str:
    """
    Remove 'References' section if it exists in extracted text.
    Prevents duplicate citations during processing.
    """
    pattern = re.compile(r"\bReferences\b.*", re.IGNORECASE | re.DOTALL)
    return re.sub(pattern, "", text).strip()
