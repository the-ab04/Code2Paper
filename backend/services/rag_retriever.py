# backend/services/rag_retriever.py

import os
from typing import List, Dict, Any, Union
from uuid import uuid4

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

# === Initialize clients ===
# Use url + optional API key for modern QdrantClient initialization
if QDRANT_API_KEY:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(url=QDRANT_URL)

# Sentence transformer embedder
try:
    embedder = SentenceTransformer(EMBED_MODEL)
except Exception as e:
    # If model cannot be loaded, raise early but print helpful message
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


# === Indexing ===
def index_paper(paper: Dict[str, Any]) -> None:
    """
    Extract text from PDF, chunk, embed, and push into Qdrant.

    Expected paper dict:
        { "title", "authors", "year", "doi", "pdf_path" }

    The payload includes both "chunk" and "text" keys for compatibility.
    """
    pdf_path = paper.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[Indexing Skipped] No PDF found for {paper.get('title')}")
        return

    ensure_collection()

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[Indexing Skipped] Empty text for {paper.get('title')}")
        return

    chunks = chunk_text(text)
    if not chunks:
        print(f"[Indexing Skipped] No chunks for {paper.get('title')}")
        return

    try:
        vectors = embedder.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        print(f"[Embedding Error] {paper.get('title')}: {e}")
        return

    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "title": paper.get("title"),
            "doi": paper.get("doi"),
            "year": paper.get("year"),
            "authors": paper.get("authors"),
            # store both keys to ensure the retrieval returns "text"
            "chunk": chunk,
            "text": chunk,
            "chunk_id": idx,
        }
        points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))

    try:
        # Upsert in batches if large
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"✅ Indexed {len(points)} chunks from {paper.get('title')}")
    except Exception as e:
        print(f"[Qdrant Error] upsert failed: {e}")


# === Retrieval ===
def query_chunks(
    queries: Union[str, List[str]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for most relevant chunks given one or multiple queries.
    Always returns items that include a 'text' key (string).
    """
    ensure_collection()

    if isinstance(queries, str):
        queries = [queries]

    results_all: List[Dict[str, Any]] = []

    for query in queries:
        try:
            # embed the query
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
            # Prefer 'text' payload key, fall back to 'chunk' or empty string
            chunk_text_val = payload.get("text") or payload.get("chunk") or ""
            results_all.append(
                {
                    "title": payload.get("title"),
                    "doi": payload.get("doi"),
                    "year": payload.get("year"),
                    "authors": payload.get("authors"),
                    "text": chunk_text_val,  # canonical 'text' key guaranteed
                    "score": getattr(hit, "score", None),
                    "query": query,
                }
            )

    # Sort by score descending (None scores go to end)
    results_all.sort(key=lambda r: (r["score"] is not None, r["score"]), reverse=True)

    # Deduplicate by (doi, snippet start) to avoid near-duplicates
    seen = set()
    unique_results: List[Dict[str, Any]] = []
    for r in results_all:
        doi = r.get("doi")
        snippet = (r.get("text") or "")[:120]
        key = (doi, snippet)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
        if len(unique_results) >= top_k:
            break

    return unique_results
