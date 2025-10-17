# backend/services/rag_retriever.py

import os
from typing import List, Dict, Any, Union, Optional
from uuid import uuid4
import math

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# === Config ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # optional
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "code2paper_chunks")

# Embedding model (SentenceTransformers)
EMBED_MODEL = os.getenv("EMBED_MODEL", "krlvi/sentence-t5-base-nlpl-code_search_net")
BATCH_UPSERT_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH", 128))

# === Initialize clients ===
if QDRANT_API_KEY:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(url=QDRANT_URL)

try:
    embedder = SentenceTransformer(EMBED_MODEL)
except Exception as e:
    raise RuntimeError(f"[Embedder Error] Failed to load embedding model '{EMBED_MODEL}': {e}")


# === Helpers ===
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file. Returns empty string on failure."""
    text_parts: List[str] = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        print(f"[PDF Error] {pdf_path}: {e}")
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks (word-based)."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words).strip()]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def ensure_collection():
    """Create collection if it doesn’t exist. Uses embedder dimension automatically."""
    try:
        if not qdrant.collection_exists(QDRANT_COLLECTION):
            vec_size = embedder.get_sentence_embedding_dimension()
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
            print(f"✅ Created Qdrant collection: {QDRANT_COLLECTION} (dim={vec_size})")
    except Exception as e:
        print(f"[Qdrant Error] ensure_collection failed: {e}")


def _paper_has_required_metadata(paper: Dict[str, Any]) -> bool:
    """
    Decide whether a paper has sufficient metadata to be indexed.
    Required: title, authors, year, doi (or venue) and local pdf_path exists.
    """
    if not paper:
        return False
    title = paper.get("title")
    authors = paper.get("authors")
    year = paper.get("year")
    doi = paper.get("doi") or paper.get("DOI")
    venue = paper.get("venue") or paper.get("container") or paper.get("container-title")
    pdf_path = paper.get("pdf_path")

    if not title or not str(title).strip():
        return False
    if not authors or not str(authors).strip():
        return False
    if not year:
        return False
    # year should be parseable as int or contains 4-digit year
    try:
        if isinstance(year, str):
            if not any(ch.isdigit() for ch in year):
                return False
        else:
            int(year)
    except Exception:
        return False

    # prefer DOI for reliable referencing; if not present, venue must exist
    if not doi and not venue:
        return False

    if not pdf_path or not os.path.exists(pdf_path):
        return False

    return True


# === Indexing ===
def index_paper(paper: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> None:
    """
    Extract text from PDF, chunk, embed, and push into Qdrant.

    Expected paper dict:
        { "title", "authors", "year", "doi" (optional), "pdf_path", "id" or "paper_id" (optional) }

    Only indexes if paper passes _paper_has_required_metadata().
    The payload will include:
      - title, doi, year, authors, paper_id (if provided), chunk, chunk_id, text
    """
    pdf_path = paper.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[Indexing Skipped] No PDF found for {paper.get('title')}")
        return

    # Only index sufficiently complete papers
    if not _paper_has_required_metadata(paper):
        print(f"[Indexing Skipped] Insufficient metadata for '{paper.get('title')}' (skipping)")
        return

    ensure_collection()

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[Indexing Skipped] Empty text for {paper.get('title')}")
        return

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        print(f"[Indexing Skipped] No chunks for {paper.get('title')}")
        return

    # Compute embeddings in batches to avoid memory blowups
    vectors = []
    try:
        # embedder.encode accepts list; chunk into manageable batches if large
        batch = BATCH_UPSERT_SIZE
        for i in range(0, len(chunks), batch):
            sub = chunks[i : i + batch]
            vecs = embedder.encode(sub, show_progress_bar=False)
            # ensure we have list of vectors
            vectors.extend([v.tolist() if hasattr(v, "tolist") else v for v in vecs])
    except Exception as e:
        print(f"[Embedding Error] {paper.get('title')}: {e}")
        return

    # Prepare points and upsert in batches
    points: List[PointStruct] = []
    paper_id = paper.get("id") or paper.get("paper_id")  # DB id if provided
    doi = paper.get("doi") or paper.get("DOI")
    title = paper.get("title")
    authors = paper.get("authors")
    year = paper.get("year")

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "title": title,
            "doi": doi,
            "year": year,
            "authors": authors,
            "chunk": chunk,
            "text": chunk,
            "chunk_id": idx,
        }
        if paper_id:
            payload["paper_id"] = str(paper_id)
        # Unique id per point ensures upsert creates new points
        points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))

    # Upsert in batches
    try:
        total = len(points)
        if total == 0:
            print(f"[Indexing] No points to upsert for {title}")
            return
        batch = BATCH_UPSERT_SIZE
        for i in range(0, total, batch):
            batch_points = points[i : i + batch]
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
        print(f"✅ Indexed {len(points)} chunks from '{title}' (paper_id={paper_id} doi={doi})")
    except Exception as e:
        print(f"[Qdrant Error] upsert failed for '{title}': {e}")


# === Retrieval ===
def query_chunks(
    queries: Union[str, List[str]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for most relevant chunks given one or multiple queries.
    Always returns items that include:
      - 'text' (the chunk text),
      - 'doi' (if present),
      - 'paper_id' (if present),
      - 'score' and 'query'.
    """
    ensure_collection()

    if isinstance(queries, str):
        queries = [queries]

    results_all: List[Dict[str, Any]] = []

    for query in queries:
        try:
            q_vec = embedder.encode([query], show_progress_bar=False)[0].tolist()
        except Exception as e:
            print(f"[Embedding Error] Failed to embed query '{query}': {e}")
            continue

        try:
            hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=top_k)
        except Exception as e:
            print(f"[Qdrant Error] search failed for query '{query}': {e}")
            continue

        for hit in hits:
            payload = hit.payload or {}
            chunk_text_val = payload.get("text") or payload.get("chunk") or ""
            results_all.append(
                {
                    "title": payload.get("title"),
                    "doi": payload.get("doi"),
                    "paper_id": payload.get("paper_id"),
                    "year": payload.get("year"),
                    "authors": payload.get("authors"),
                    "text": chunk_text_val,
                    "score": getattr(hit, "score", None),
                    "query": query,
                }
            )

    # Sort by score descending (None scores to end)
    results_all.sort(key=lambda r: (r["score"] is not None, r["score"]), reverse=True)

    # Deduplicate by (doi, snippet start) to avoid near-duplicates
    seen = set()
    unique_results: List[Dict[str, Any]] = []
    for r in results_all:
        doi = r.get("doi")
        snippet = (r.get("text") or "")[:160]
        key = (doi, snippet)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
        if len(unique_results) >= top_k:
            break

    return unique_results
