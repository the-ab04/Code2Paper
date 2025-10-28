# backend/services/paper_finder.py
"""
Paper discovery helper.

Improvements / behavior in this updated version:
  - Tunables at top (MAX_METADATA_RESULTS, MAX_PDF_DOWNLOADS, MIN_YEAR) control result sizes.
  - Metadata-first approach: collect candidates from CrossRef + arXiv, dedupe, score & filter.
  - Only attempt PDF downloads for the top-K candidates (limits network/disk/time).
  - Avoid overwriting existing PDF files by appending a short deterministic suffix (from DOI/title).
  - Best-effort lightweight indexing for downloaded PDFs (calls index_paper with index_chunks=False if supported).
  - Robust error handling and informative prints for debugging.
  - Query expansion using detected model names, dataset names and metric tokens.
  - Fuzzy title deduplication to avoid near-duplicate titles.
  - Query deduplication and semantic normalization to avoid repeated identical searches.
"""
import os
import re
import requests
import difflib
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import quote_plus
from datetime import datetime
import hashlib

from services.rag_retriever import index_paper  # auto-index PDFs into Qdrant
from db import crud, schemas

# === Config ===
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

# Unpaywall requires a registered email (set in .env)
UNPAYWALL_EMAIL = 'temp@gmail.com'

# Central storage for PDFs
PAPER_STORAGE = os.path.join("storage", "papers")
os.makedirs(PAPER_STORAGE, exist_ok=True)

# Default request headers (help CrossRef & others treat you politely)
DEFAULT_HEADERS = {
    "User-Agent": "Code2Paper/1.0 (mailto:your-real-email@example.com)"
}

# Tunables (via environment)
MAX_METADATA_RESULTS = int(os.getenv("PAPERFINDER_MAX_METADATA", "20"))  # per query (cap)
MAX_PDF_DOWNLOADS = int(os.getenv("PAPERFINDER_MAX_PDF_DOWNLOADS", "6"))  # per query
MIN_YEAR = int(os.getenv("PAPERFINDER_MIN_YEAR", "2015"))  # prefer newer than this year

# Fuzzy-title dedupe threshold (0..1)
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("PAPERFINDER_TITLE_SIMILARITY", "0.86"))

# HTTP request session (reuse connections)
_SESSION = requests.Session()
_SESSION.headers.update(DEFAULT_HEADERS)


# === Utilities ===
def clean_filename(name: str) -> str:
    """Make a safe filename for saving PDFs."""
    if not name:
        name = "paper"
    # Keep some punctuation stripped; keep ASCII alphanum, dash, underscore
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    return name[:200]


def _normalize_title_key(title: Optional[str]) -> str:
    if not title:
        return ""
    t = re.sub(r"\s+", " ", title).strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    return t[:240]


def _is_similar_title(a: str, b: str, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Return True if titles 'a' and 'b' look similar (fuzzy match)."""
    if not a or not b:
        return False
    a_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", a.lower())).strip()
    b_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", b.lower())).strip()
    if not a_norm or not b_norm:
        return False
    # quick token overlap check (fast)
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if a_tokens and b_tokens:
        overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
        if overlap >= 0.6:
            return True
    # fallback to sequence matcher ratio
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold


def is_candidate(paper: Dict[str, Any]) -> bool:
    """
    Determine if paper is a usable candidate for citing/persisting.
    Minimal requirements: title and authors present (non-empty).
    """
    if not paper:
        return False
    title = paper.get("title")
    authors = paper.get("authors")
    if not title or not str(title).strip():
        return False
    if not authors or not str(authors).strip():
        return False
    return True


def is_indexable(paper: Dict[str, Any]) -> bool:
    """
    Determine whether the paper can be indexed into Qdrant (requires local PDF).
    """
    pdf_path = paper.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        return False
    return is_candidate(paper)


def _parse_year(val: Any) -> Optional[int]:
    """Extract 4-digit year from various value types (string or int)."""
    if val is None:
        return None
    try:
        if isinstance(val, int):
            return val
        s = str(val)
        m = re.search(r"(\d{4})", s)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _short_suffix_from_string(s: str, length: int = 8) -> str:
    """Return a short deterministic hex suffix for given string (used to avoid filename collisions)."""
    if not s:
        return ""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:length]


# common lists for query expansion (can be extended)
_COMMON_MODEL_KEYWORDS = [
    "cnn", "convnet", "resnet", "unet", "transformer", "bert", "gpt", "lstm", "gru", "hybrid", "darts",
    "random_forest", "xgboost", "lightgbm", "svm", "mlp"
]
_COMMON_DATASET_KEYWORDS = [
    "cifar", "mnist", "imagenet", "uciadult", "uci", "kitti", "cityscapes", "wikipedia", "squad",
    "wmt", "imdb", "coco", "kaggle", "time_series", "electricity", "bitcoin", "ethereum"
]
_COMMON_METRIC_KEYWORDS = ["accuracy", "acc", "loss", "mse", "rmse", "f1", "precision", "recall", "auc"]


def _extract_tokens_for_expansion(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Heuristic extraction of model names, dataset tokens, and metric tokens from text.
    Returns (models, datasets, metrics).
    """
    if not text:
        return [], [], []
    text_l = text.lower()
    models = [m for m in _COMMON_MODEL_KEYWORDS if m in text_l]
    datasets = [d for d in _COMMON_DATASET_KEYWORDS if d in text_l]
    metrics = [m for m in _COMMON_METRIC_KEYWORDS if m in text_l]
    # also try to extract simple CamelCase or ModelClass names (e.g., HybridCNNTransformer)
    model_classes = re.findall(r"\b([A-Z][a-zA-Z]{3,40}(?:Model|Net|Transformer|CNN|RNN|LSTM|GRU)?)\b", text)
    for mc in model_classes:
        mc_lower = mc.lower()
        if mc_lower not in models:
            models.append(mc_lower)
    return models, datasets, metrics


def _expand_queries(orig_queries: List[str], max_expansions_per_query: int = 4) -> List[str]:
    """
    For each seed query, generate a small set of expanded queries that combine
    extracted model/dataset/metric tokens. Keep the total expansions bounded.
    """
    if not orig_queries:
        return []
    expanded = []
    for q in orig_queries:
        expanded.append(q)
        models, datasets, metrics = _extract_tokens_for_expansion(q)
        combos = []
        # prefer combos: model + dataset, model + metric, dataset + metric
        for m in models[:2]:
            for d in datasets[:2]:
                combos.append(f"{m} {d}")
            for met in metrics[:2]:
                combos.append(f"{m} {met}")
        for d in datasets[:2]:
            for met in metrics[:2]:
                combos.append(f"{d} {met}")
        # also add model alone and dataset alone if found
        combos.extend(models[:2])
        combos.extend(datasets[:2])
        # limit and attach original query context
        added = 0
        for c in combos:
            if added >= max_expansions_per_query:
                break
            c_full = f"{q} {c}"
            if c_full not in expanded:
                expanded.append(c_full)
                added += 1
    # dedupe while preserving order
    seen = set()
    out = []
    for e in expanded:
        k = e.strip().lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


# --- NEW: semantic normalization for queries (to collapse tiny variants) ---
def normalize_query_semantically(q: str) -> str:
    """
    Return a canonicalized string for a query to allow semantic deduplication.
    - Lowercases, collapses common metric abbreviations: 'acc' -> 'accuracy'
    - Normalizes common whitespace/punctuation
    - Collapses synonyms for model/dataset tokens if possible
    """
    if not q:
        return ""
    s = str(q).strip().lower()
    # normalize common metric shortcuts
    s = re.sub(r'\bacc\b', 'accuracy', s)
    s = re.sub(r'\baccu\b', 'accuracy', s)
    s = re.sub(r'\bfigs?\b', 'figure', s)
    # collapse multiple separators
    s = re.sub(r'[_\-\+\/]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    # keep only alnum and spaces for normalization key
    s = re.sub(r'[^\w\s]', '', s)
    return s


# === CrossRef Search ===
def search_crossref(query: str, rows: int = 5) -> List[Dict[str, Any]]:
    """Search CrossRef for papers matching query and normalize metadata."""
    try:
        rows = max(1, int(rows))
        resp = _SESSION.get(
            CROSSREF_API,
            params={"query": query, "rows": rows, "mailto": DEFAULT_HEADERS.get("User-Agent")},
            timeout=12,
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            normalized = []
            for item in items:
                title = item.get("title", [""])[0] if item.get("title") else ""
                authors_list = item.get("author", []) or []
                authors = ", ".join(
                    f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
                    for a in authors_list
                    if a.get("family") or a.get("given")
                ) or None
                year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
                container = item.get("container-title", [""])[0] if item.get("container-title") else ""
                volume = item.get("volume", "")
                issue = item.get("issue", "")
                pages = item.get("page", "")
                doi = item.get("DOI")
                url = item.get("URL")
                normalized.append(
                    {
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "venue": container,
                        "volume": volume,
                        "issue": issue,
                        "pages": pages,
                        "doi": doi,
                        "url": url,
                        "source": "crossref",
                    }
                )
            return normalized
        else:
            print(f"[CrossRef] non-200 status {resp.status_code} for query '{query}'")
    except Exception as e:
        print(f"[CrossRef Error] {e}")
    return []


# === ArXiv Search ===
def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv API for papers. Note: many arXiv results lack DOI/venue; they will be returned as candidates."""
    import feedparser

    try:
        url = f"{ARXIV_API}?search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            authors = ", ".join(a.name for a in getattr(entry, "authors", [])) if getattr(entry, "authors", None) else None
            year = getattr(entry, "published", "").split("-")[0] if getattr(entry, "published", "") else None
            # arXiv sometimes supplies DOI mapping fields
            doi = entry.get("arxiv_doi") or entry.get("doi") or None
            pdf_url = next(
                (l.href for l in entry.links if getattr(l, "type", "") == "application/pdf"), None
            )
            url = entry.get("link")
            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": "arXiv",
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "doi": doi,
                    "url": url,
                    "pdf_url": pdf_url,
                    "source": "arxiv",
                }
            )
        return results
    except Exception as e:
        print(f"[arXiv Error] {e}")
        return []


# === Open Access Check via Unpaywall ===
def get_unpaywall_oa_link(doi: str) -> Optional[str]:
    """Check Unpaywall for OA full-text PDF link given a DOI."""
    if not doi or not UNPAYWALL_EMAIL:
        return None
    try:
        resp = _SESSION.get(
            f"{UNPAYWALL_API}/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=12,
        )
        if resp.status_code == 200:
            oa_location = resp.json().get("best_oa_location")
            if oa_location and oa_location.get("url_for_pdf"):
                return oa_location["url_for_pdf"]
    except Exception as e:
        print(f"[Unpaywall Error] {e}")
    return None


# === PDF Downloader ===
def _unique_pdf_path(title: str, doi: Optional[str] = None) -> str:
    """Return a unique filepath for saving PDF (avoid collisions)."""
    base = clean_filename(title or "paper")
    suffix = ""
    if doi:
        suffix = _short_suffix_from_string(doi, length=8)
    else:
        suffix = _short_suffix_from_string(title or "", length=8)
    filename = f"{base}_{suffix}.pdf"
    filepath = os.path.join(PAPER_STORAGE, filename)
    return filepath


def download_pdf(url: str, title: str, doi: Optional[str] = None) -> Optional[str]:
    """Download PDF from URL into storage/papers and return local path (unique filename)."""
    try:
        resp = _SESSION.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            # choose a deterministic filename that avoids overwriting different papers with same title
            filepath = _unique_pdf_path(title, doi)
            # If a file already exists with identical content, we could check ETag or size, but keep it simple:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(1024):
                    if not chunk:
                        continue
                    f.write(chunk)
            return filepath
        else:
            print(f"[Download] non-200 status {resp.status_code} for PDF URL {url}")
    except Exception as e:
        print(f"[Download Error] {e} for URL {url}")
    return None


# === Main: search-only (returns candidate papers, prefer those with PDF for indexing) ===
def find_papers(queries: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Given search queries (strings), return metadata + local PDF paths (if OA available).

    Steps:
      1. Expand queries heuristically (model/dataset/metric tokens).
      2. Collect metadata candidates from CrossRef and arXiv (capped).
      3. Deduplicate by DOI or normalized title + fuzzy title match.
      4. Filter by MIN_YEAR (lenient) and score (favor DOI/recent).
      5. Attempt PDF downloads only for top-K candidates (controls network/disk/time).
      6. Index only pdf-backed entries (lightweight paper-level indexing if supported).
    """
    papers: List[Dict[str, Any]] = []
    seen_keys = set()

    if not queries:
        return []

    # Expand queries to improve recall
    expanded_queries = _expand_queries(queries)
    # Cap number of queries to avoid explosion
    MAX_QUERY_EXPANSIONS = int(os.getenv("PAPERFINDER_MAX_QUERY_EXPANSIONS", "12"))
    if len(expanded_queries) > MAX_QUERY_EXPANSIONS:
        expanded_queries = expanded_queries[:MAX_QUERY_EXPANSIONS]

    # --- Deduplicate expanded queries semantically & preserve order ---
    seen_norm = set()
    normalized_expanded = []
    for e in expanded_queries:
        k = normalize_query_semantically(e)
        if not k:
            continue
        if k not in seen_norm:
            seen_norm.add(k)
            normalized_expanded.append(e)
    expanded_queries = normalized_expanded

    print(f"[Paper Finder] Original queries: {queries}")
    print(f"[Paper Finder] Expanded queries (capped to {MAX_QUERY_EXPANSIONS}): {expanded_queries}")

    # To avoid repeating the exact same search multiple times during the loop
    seen_searches = set()

    for query in expanded_queries:
        if not query or not str(query).strip():
            continue
        q = str(query).strip()

        # semantic normalized key for search skip
        q_key = normalize_query_semantically(q)
        if not q_key:
            continue
        if q_key in seen_searches:
            # already searched a semantically identical variant
            print(f"[Paper Finder] Skipping duplicate/semantically-similar search: '{q}'")
            continue
        seen_searches.add(q_key)

        print(f"[Paper Finder] Searching for query: {q}")

        # 1) Collect metadata candidates from both sources (limit to MAX_METADATA_RESULTS)
        candidates: List[Dict[str, Any]] = []
        try:
            crossref_results = search_crossref(q, rows=min(max_results, MAX_METADATA_RESULTS))
            candidates.extend(crossref_results)
        except Exception as e:
            print(f"[Paper Finder] CrossRef failed for '{q}': {e}")

        try:
            arxiv_results = search_arxiv(q, max_results=min(max_results, MAX_METADATA_RESULTS))
            candidates.extend(arxiv_results)
        except Exception as e:
            print(f"[Paper Finder] arXiv failed for '{q}': {e}")

        if not candidates:
            continue

        # 2) Normalize and deduplicate by DOI (preferred) or title key, with fuzzy title merging
        uniq: Dict[str, Dict[str, Any]] = {}
        title_index: List[Tuple[str, str]] = []  # list of (norm_title, key) for fuzzy checks
        for p in candidates:
            doi = (p.get("doi") or "") or ""
            if doi:
                key = f"doi::{str(doi).strip().lower()}"
                if key in uniq:
                    # merge missing fields
                    exist = uniq[key]
                    for k, v in p.items():
                        if (not exist.get(k)) and v:
                            exist[k] = v
                    continue
                uniq[key] = p
                title_index.append((_normalize_title_key(p.get("title")), key))
            else:
                # normalize title
                norm = _normalize_title_key(p.get("title"))
                # attempt to detect fuzzy duplicate against existing titles in uniq
                found_similar = None
                for existing_norm, existing_key in title_index:
                    if existing_norm and norm and _is_similar_title(existing_norm, norm):
                        found_similar = existing_key
                        break
                if found_similar:
                    exist = uniq[found_similar]
                    for k, v in p.items():
                        if (not exist.get(k)) and v:
                            exist[k] = v
                    continue
                # new title-based key
                key = f"title::{norm}"
                uniq[key] = p
                title_index.append((norm, key))

        deduped = list(uniq.values())

        # 3) Filter by MIN_YEAR (lenient) and then sort: prefer DOI + recent year
        now_year = datetime.now().year

        def score_candidate(p: Dict[str, Any]) -> float:
            s = 0.0
            if p.get("doi"):
                s += 2.0
            yr = _parse_year(p.get("year"))
            if yr:
                # favor recent years (normalized)
                s += max(0.0, 1.0 - ((now_year - yr) / 20.0))
            # slight preference for CrossRef (formal metadata)
            if p.get("source") == "crossref":
                s += 0.1
            # prefer entries that include a PDF link
            if p.get("pdf_url") or (p.get("url") and str(p.get("url")).lower().endswith(".pdf")):
                s += 0.2
            return s

        filtered = []
        for p in deduped:
            yr = _parse_year(p.get("year"))
            # skip very old if year present and below MIN_YEAR
            if yr is not None and yr < MIN_YEAR:
                continue
            if is_candidate(p):
                filtered.append(p)

        if not filtered:
            continue

        # sort by score desc
        filtered.sort(key=lambda x: score_candidate(x), reverse=True)

        # 4) Only attempt PDF download for top-K candidates (limits network + disk)
        to_download = filtered[:MAX_PDF_DOWNLOADS]
        remaining = filtered[MAX_PDF_DOWNLOADS:]

        downloaded_count = 0
        for p in to_download:
            doi = p.get("doi")
            pdf_path = None
            # Try Unpaywall if DOI is available
            if doi:
                try:
                    oa_link = get_unpaywall_oa_link(doi)
                except Exception:
                    oa_link = None
                if oa_link:
                    pdf_path = download_pdf(oa_link, p.get("title", "paper"), doi)
            # else try arXiv PDF if present
            if not pdf_path and p.get("pdf_url"):
                pdf_path = download_pdf(p.get("pdf_url"), p.get("title", "paper"), doi)

            if pdf_path:
                p["pdf_path"] = pdf_path
                downloaded_count += 1
                # index only PDF-backed entries (use lightweight paper-level indexing if supported)
                try:
                    # prefer calling index_paper with index_chunks=False (best-effort)
                    try:
                        index_paper(dict(p, pdf_path=pdf_path), index_chunks=False)
                    except TypeError:
                        # fallback if index_paper signature does not accept index_chunks
                        index_paper(dict(p, pdf_path=pdf_path))
                except Exception as e:
                    print(f"[Index Error] while indexing downloaded paper '{p.get('title')}': {e}")

            # Append the candidate regardless of download success (so caller can decide)
            key = (str(p.get("doi")).lower().strip() if p.get("doi") else _normalize_title_key(p.get("title")))
            # also guard against fuzzy duplicates in the 'papers' output
            already_similar = False
            if not p.get("doi"):
                for existing in papers:
                    if _is_similar_title(existing.get("title", ""), p.get("title", "")):
                        already_similar = True
                        break
            if key not in seen_keys and not already_similar:
                seen_keys.add(key)
                papers.append(p)
            else:
                if already_similar:
                    print(f"[Paper Finder] Skipped near-duplicate (fuzzy) title: {p.get('title')}")

        # 5) Append remaining filtered candidates (without downloads/index) until overall dedupe
        for p in remaining:
            key = (str(p.get("doi")).lower().strip() if p.get("doi") else _normalize_title_key(p.get("title")))
            already_similar = False
            if not p.get("doi"):
                for existing in papers:
                    if _is_similar_title(existing.get("title", ""), p.get("title", "")):
                        already_similar = True
                        break
            if key not in seen_keys and not already_similar:
                seen_keys.add(key)
                papers.append(p)
            else:
                if already_similar:
                    print(f"[Paper Finder] Skipped remaining near-duplicate title: {p.get('title')}")

        print(f"[Paper Finder] Query '{q}' -> candidates found: {len(filtered)}, downloaded: {downloaded_count}")

    # Return the discovered candidates (pdf_path may be present or None)
    return papers


# === New: find AND persist into Postgres + index with DB linkage ===
def find_and_persist_papers(
    queries: List[str],
    run_id: int,
    db,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Full pipeline helper:
      - find candidate papers (metadata-first)
      - download OA PDF (top-K)
      - persist to Postgres via crud.add_paper (linked to run_id),
      - index into Qdrant only for pdf-backed entries (best-effort).

    Returns list of persisted paper dicts (each includes DB id via 'db_id' key and 'pdf_path' maybe None).
    """
    persisted: List[Dict[str, Any]] = []

    discovered = find_papers(queries, max_results=max_results)

    for paper in discovered:
        # Minimal check before persisting: title & authors
        if not is_candidate(paper):
            continue

        # Safely coerce year to int if possible, else None
        year_val = None
        try:
            if paper.get("year") is not None:
                y = str(paper.get("year")).strip()
                m = re.search(r"(\d{4})", y)
                if m:
                    year_val = int(m.group(1))
                else:
                    year_val = None
        except Exception:
            year_val = None

        # Build PaperBase schema for DB insertion
        paper_base = schemas.PaperBase(
            title=paper.get("title") or "Untitled",
            authors=paper.get("authors"),
            year=year_val,
            venue=paper.get("venue"),
            doi=paper.get("doi"),
            url=paper.get("url"),
            pdf_path=paper.get("pdf_path"),
        )

        try:
            db_paper = crud.add_paper(db, run_id, paper_base)
            if not db_paper:
                print(f"[DB Warning] crud.add_paper returned None for title='{paper.get('title')}'. Skipping persistence.")
                continue
        except Exception as e:
            print(f"[DB Error] Failed to add paper '{paper.get('title')}': {e}")
            continue

        paper_out = dict(paper)
        # guard in case db_paper is not what we expect
        try:
            paper_out["db_id"] = getattr(db_paper, "id", None)
        except Exception:
            paper_out["db_id"] = None
        paper_out["pdf_path"] = paper.get("pdf_path")

        # Index into Qdrant with DB linkage only if pdf_path available
        try:
            if paper_out.get("pdf_path") and os.path.exists(paper_out.get("pdf_path")):
                idx_payload = dict(paper_out)
                idx_payload["paper_id"] = paper_out.get("db_id")
                try:
                    index_paper(idx_payload, index_chunks=False)
                except TypeError:
                    index_paper(idx_payload)
        except Exception as e:
            print(f"[Index Error] Failed to index paper id={paper_out.get('db_id')} title='{paper.get('title')}': {e}")

        persisted.append(paper_out)

    print(f"[Paper Finder] Persisted {len(persisted)} candidate papers for run {run_id}")
    return persisted
