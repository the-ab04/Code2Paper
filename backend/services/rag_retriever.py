# backend/services/rag_retriever.py

import os
import re
from typing import List, Dict, Any, Union, Optional
from uuid import uuid4
import uuid

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
BATCH_UPSERT_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH", "128"))

# Chunking / indexing controls (tune to speed up or quality)
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "-1"))  # -1 => unlimited
RAG_MAX_PAGES = int(os.getenv("RAG_MAX_PAGES", "-1"))  # -1 => read all pages

# Minimum score threshold to accept retrieval hits (cosine similarity style; tune per-setup)
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.20"))

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
def extract_text_from_pdf(pdf_path: str, max_pages: int = RAG_MAX_PAGES) -> str:
    """Extract plain text from the first `max_pages` of a PDF file. Returns empty string on failure."""
    text_parts: List[str] = []
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        end_page = num_pages if max_pages < 0 else min(num_pages, max_pages)
        for i in range(end_page):
            try:
                page = reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                # continue despite extraction errors on single pages
                continue
    except Exception as e:
        print(f"[PDF Error] {pdf_path}: {e}")
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
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
        # stop early if we have reached RAG_MAX_CHUNKS
        if RAG_MAX_CHUNKS > 0 and len(chunks) >= RAG_MAX_CHUNKS:
            break
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
    Required: title, authors, year (lenient), and local pdf_path exists.
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
        # be lenient: allow indexing if other strong identifiers exist
        # return False
        pass
    if not pdf_path or not os.path.exists(pdf_path):
        return False
    return True


def _stable_paper_point_id(paper: Dict[str, Any]) -> Union[int, str]:
    """
    Create a stable ID for the paper-level Qdrant point so re-indexing can upsert instead of duplicating.

    Rules:
      - If DB paper id (paper.get('id') or 'paper_id') is integer-like -> return int
      - Else if DOI exists -> return deterministic UUID5 string derived from DOI (valid UUID string)
      - Else -> return a random UUID string
    """
    pid = paper.get("id") or paper.get("paper_id")
    doi = paper.get("doi") or paper.get("DOI")

    # prefer integer DB id (Qdrant accepts integer point ids)
    if pid is not None:
        try:
            pid_int = int(str(pid))
            return pid_int
        except Exception:
            pass

    # deterministic UUID from DOI (stable across re-indexing)
    if doi:
        doi_norm = str(doi).strip().lower()
        try:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, doi_norm))
        except Exception:
            return str(uuid.uuid4())

    # fallback: random uuid string
    return str(uuid.uuid4())


# === Indexing ===
def index_paper(
    paper: Dict[str, Any],
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
    index_chunks: bool = True,
) -> None:
    """
    Extract text from PDF, chunk, embed, and push into Qdrant.

    Parameters:
      - paper: dict with keys {title, authors, year, doi (optional), pdf_path, id/paper_id (optional)}
      - chunk_size, overlap: chunking behavior
      - index_chunks: if False, only upsert a single paper-level point (fast).
    
    Behavior:
      - If index_chunks is True: create chunk-level points + paper-level stable point.
      - If index_chunks is False: compute a concise summary vector and upsert only the paper-level point.
    """
    pdf_path = paper.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[Indexing Skipped] No PDF found for {paper.get('title')}")
        return

    # Only index sufficiently complete papers (for chunk indexing)
    if index_chunks and not _paper_has_required_metadata(paper):
        print(f"[Indexing Skipped] Insufficient metadata for '{paper.get('title')}' (skipping chunk indexing)")
        return

    ensure_collection()

    # Extract text (possibly limited by RAG_MAX_PAGES)
    text = extract_text_from_pdf(pdf_path, max_pages=RAG_MAX_PAGES)
    if not text.strip():
        print(f"[Indexing Skipped] Empty text for {paper.get('title')}")
        return

    title = paper.get("title")
    authors = paper.get("authors")
    year = paper.get("year")
    doi = paper.get("doi") or paper.get("DOI")
    paper_id = paper.get("id") or paper.get("paper_id")

    # If only paper-level index is requested, compute a concise summary and upsert a single point
    if not index_chunks:
        try:
            # create a compact summary using title + first ~600 chars from text
            preview = (text[:1200]).strip()
            paper_summary = f"{title or ''}. {preview}".strip()
            vec = embedder.encode([paper_summary], show_progress_bar=False)[0]
            vector = vec.tolist() if hasattr(vec, "tolist") else vec

            stable_id = _stable_paper_point_id(paper)
            payload = {
                "title": title,
                "doi": doi,
                "year": year,
                "authors": authors,
                "paper_id": str(paper_id) if paper_id else None,
                "chunk_id": -1,
                "text": preview,
            }
            point = PointStruct(id=stable_id, vector=vector, payload=payload)
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])
            print(f"✅ Paper-level indexed (summary) for '{title}' (paper_id={paper_id} doi={doi})")
        except Exception as e:
            print(f"[Qdrant Error] paper-level upsert failed for '{title}': {e}")
        return

    # --- Otherwise: embed & upsert chunks (slower) ---
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        print(f"[Indexing Skipped] No chunks for {title}")
        return

    # Compute chunk embeddings in batches
    vectors = []
    try:
        batch = BATCH_UPSERT_SIZE
        for i in range(0, len(chunks), batch):
            sub = chunks[i : i + batch]
            vecs = embedder.encode(sub, show_progress_bar=False)
            vectors.extend([v.tolist() if hasattr(v, "tolist") else v for v in vecs])
    except Exception as e:
        print(f"[Embedding Error] {title}: {e}")
        return

    # Prepare chunk points
    points: List[PointStruct] = []
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
        # unique id per chunk
        points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))

    # Also compute and upsert a paper-level vector (stable id)
    try:
        preview_text = chunks[0][:1200] if chunks else (text[:1200] if text else "")
        paper_summary = f"{title or ''}. {preview_text}".strip()
        pvec = embedder.encode([paper_summary], show_progress_bar=False)[0]
        paper_vector = pvec.tolist() if hasattr(pvec, "tolist") else pvec
        stable_id = _stable_paper_point_id(paper)
        paper_payload = {
            "title": title,
            "doi": doi,
            "year": year,
            "authors": authors,
            "paper_id": str(paper_id) if paper_id else None,
            "chunk_id": -1,
            "text": preview_text,
        }
        paper_point = PointStruct(id=stable_id, vector=paper_vector, payload=paper_payload)
    except Exception as e:
        print(f"[Embedding Error] paper-level embed failed for '{title}': {e}")
        paper_point = None

    # Upsert in batches
    try:
        total = len(points)
        if total > 0:
            batch = BATCH_UPSERT_SIZE
            for i in range(0, total, batch):
                batch_points = points[i : i + batch]
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
        if paper_point:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[paper_point])
        print(f"✅ Indexed {len(points)} chunks from '{title}' (paper_id={paper_id} doi={doi})")
    except Exception as e:
        print(f"[Qdrant Error] upsert failed for '{title}': {e}")


# === Retrieval ===
def query_chunks(
    queries: Union[str, List[str]],
    top_k: int = 5,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for most relevant chunks given one or multiple queries.
    Returns items with:
      - 'text' (the chunk text),
      - 'doi' (if present),
      - 'paper_id' (if present),
      - 'score' and 'query'.

    min_score: float threshold below which hits are omitted. Defaults to RETRIEVAL_SCORE_THRESHOLD.
    """
    ensure_collection()

    if isinstance(queries, str):
        queries = [queries]

    if min_score is None:
        min_score = RETRIEVAL_SCORE_THRESHOLD

    results_all: List[Dict[str, Any]] = []

    for query in queries:
        try:
            q_vec = embedder.encode([query], show_progress_bar=False)[0].tolist()
        except Exception as e:
            print(f"[Embedding Error] Failed to embed query '{query}': {e}")
            continue

        try:
            hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=top_k * 3)
        except Exception as e:
            print(f"[Qdrant Error] search failed for query '{query}': {e}")
            continue

        for hit in hits:
            score = getattr(hit, "score", None)
            try:
                if score is not None and min_score is not None and score < min_score:
                    continue
            except Exception:
                pass

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
                    "score": score,
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


def query_papers_by_summary(summary: str, top_k: int = 5, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Search Qdrant using a short summary (facts_summary) and return best-matching papers (paper-level aggregation).
    Uses the paper-level points (chunk_id = -1) and/or aggregates chunk hits to compute a per-paper score.

    Returns list of dicts:
      { 'paper_id', 'doi', 'title', 'authors', 'year', 'score', 'sample_text' }
    """
    ensure_collection()

    if not summary or not str(summary).strip():
        return []

    if min_score is None:
        min_score = RETRIEVAL_SCORE_THRESHOLD

    try:
        q_vec = embedder.encode([summary], show_progress_bar=False)[0].tolist()
    except Exception as e:
        print(f"[Embedding Error] Failed to embed summary: {e}")
        return []

    try:
        # Search a larger result set so we can aggregate per-paper
        hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=max(50, top_k * 5))
    except Exception as e:
        print(f"[Qdrant Error] paper-level search failed: {e}")
        return []

    # Aggregate by paper identifier (prefer paper_id, else doi, else title)
    per_paper: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        score = getattr(hit, "score", None)
        if score is not None and min_score is not None and score < min_score:
            continue

        payload = hit.payload or {}
        paper_id = payload.get("paper_id") or payload.get("paperId") or None
        doi = payload.get("doi")
        title = payload.get("title") or ""
        authors = payload.get("authors")
        year = payload.get("year")
        text_snippet = payload.get("text") or payload.get("chunk") or ""

        # Build stable key
        key = None
        if paper_id:
            key = f"pid_{paper_id}"
        elif doi:
            key = f"doi_{str(doi).strip().lower()}"
        else:
            clean_title = re.sub(r'\s+', '_', (title or '').strip().lower())
            key = f"title_{clean_title[:120]}"

        if key not in per_paper:
            per_paper[key] = {
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "authors": authors,
                "year": year,
                "score": score if score is not None else 0.0,
                "sample_text": text_snippet,
            }
        else:
            existing = per_paper[key]
            if score is not None and (existing.get("score") is None or score > existing.get("score", 0.0)):
                existing["score"] = score
                existing["sample_text"] = text_snippet or existing.get("sample_text")

    candidates = list(per_paper.values())
    candidates = [c for c in candidates if (c.get("score") is None or c.get("score") >= min_score)]
    candidates.sort(key=lambda x: (x.get("score") is not None, x.get("score")), reverse=True)

    return candidates[:top_k]
