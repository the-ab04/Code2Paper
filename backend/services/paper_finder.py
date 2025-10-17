# backend/services/paper_finder.py

import os
import re
import requests
from typing import List, Dict, Optional, Any

from services.rag_retriever import index_paper  # auto-index PDFs into Qdrant
from db import crud, schemas

# === Config ===
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

# Unpaywall requires a registered email (set in .env)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

# Central storage for PDFs
PAPER_STORAGE = os.path.join("storage", "papers")
os.makedirs(PAPER_STORAGE, exist_ok=True)

# Default request headers (help CrossRef & others treat you politely)
DEFAULT_HEADERS = {
    "User-Agent": "Code2Paper/1.0 (mailto:your-real-email@example.com)"
}


# === Utilities ===
def clean_filename(name: str) -> str:
    """Make a safe filename for saving PDFs."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)[:200]


def is_complete(paper: Dict[str, Any]) -> bool:
    """
    Check whether the paper has all required fields for an IEEE-style reference.
    Required fields: title, authors, year, venue, doi.
    Note: We require a downloaded pdf_path elsewhere (so downstream we only index papers with pdfs).
    """
    required = ["title", "authors", "year", "venue", "doi"]
    for k in required:
        val = paper.get(k)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False

    # year should contain a 4-digit year (lenient)
    year = paper.get("year")
    try:
        if isinstance(year, str):
            if not re.search(r"\d{4}", year):
                return False
        else:
            int(year)
    except Exception:
        return False
    return True


# === CrossRef Search ===
def search_crossref(query: str, rows: int = 5) -> List[Dict[str, Any]]:
    """Search CrossRef for papers matching query and normalize metadata."""
    try:
        resp = requests.get(
            CROSSREF_API,
            params={"query": query, "rows": rows, "mailto": DEFAULT_HEADERS.get("User-Agent")},
            headers=DEFAULT_HEADERS,
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
    """Search arXiv API for papers. Note: many arXiv results lack DOI/venue; they will be filtered out."""
    import feedparser

    url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            authors = ", ".join(a.name for a in getattr(entry, "authors", [])) if getattr(entry, "authors", None) else None
            year = getattr(entry, "published", "").split("-")[0] if getattr(entry, "published", "") else None
            # arXiv sometimes supplies arXiv DOI mapping in different fields
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
                    "venue": "arXiv",  # explicit - we'll still require DOI to mark complete
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
        resp = requests.get(
            f"{UNPAYWALL_API}/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            headers=DEFAULT_HEADERS,
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
def download_pdf(url: str, title: str) -> Optional[str]:
    """Download PDF from URL into storage/papers and return local path."""
    try:
        resp = requests.get(url, stream=True, timeout=20, headers=DEFAULT_HEADERS)
        if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
            filename = clean_filename(title) + ".pdf"
            filepath = os.path.join(PAPER_STORAGE, filename)
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return filepath
        else:
            # Some PDF endpoints don't set content-type correctly; attempt a relaxed save if status 200
            if resp.status_code == 200:
                filename = clean_filename(title) + ".pdf"
                filepath = os.path.join(PAPER_STORAGE, filename)
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)
                # note: we still return it but indexing may fail if PDF is malformed
                return filepath
    except Exception as e:
        print(f"[Download Error] {e}")
    return None


# === Main: search-only (returns complete papers only) ===
def find_papers(queries: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Given search queries, return metadata + local PDF paths (if OA available).
    Auto-indexes PDFs into Qdrant only for papers that contain all required fields
    for IEEE-style referencing (title, authors, year, venue, doi).
    This function does NOT persist to DB; it returns the verified metadata list.
    """
    papers: List[Dict[str, Any]] = []
    seen_keys = set()

    for query in queries:
        # CrossRef
        crossref_results = search_crossref(query, rows=max_results)
        for paper in crossref_results:
            doi = paper.get("doi")
            # Try to download OA PDF via Unpaywall if DOI exists
            if doi:
                oa_link = get_unpaywall_oa_link(doi)
            else:
                oa_link = None

            if oa_link:
                pdf_path = download_pdf(oa_link, paper.get("title", "paper"))
                if pdf_path:
                    paper["pdf_path"] = pdf_path

            # Only accept if complete metadata and PDF is present
            if is_complete(paper) and paper.get("pdf_path"):
                key = (str(paper.get("doi")).lower().strip() if paper.get("doi") else paper.get("title"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    # Index into Qdrant (best-effort)
                    try:
                        index_paper(paper)
                    except Exception as e:
                        print(f"[Index Error] {e}")
                    papers.append(paper)

        # arXiv
        arxiv_results = search_arxiv(query, max_results=max_results)
        for paper in arxiv_results:
            pdf_url = paper.get("pdf_url")
            if pdf_url:
                pdf_path = download_pdf(pdf_url, paper.get("title", "paper"))
                if pdf_path:
                    paper["pdf_path"] = pdf_path

            if is_complete(paper) and paper.get("pdf_path"):
                key = (str(paper.get("doi")).lower().strip() if paper.get("doi") else paper.get("title"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    try:
                        index_paper(paper)
                    except Exception as e:
                        print(f"[Index Error] {e}")
                    papers.append(paper)

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
      - find candidate papers,
      - download OA PDF (if available),
      - ensure required metadata present (is_complete),
      - persist to Postgres via crud.add_paper (linked to run_id),
      - index into Qdrant (include paper_id in payload)
    Returns list of persisted paper dicts (each includes DB id via 'db_id' key and 'pdf_path').
    """
    persisted: List[Dict[str, Any]] = []

    # Use find_papers-like discovery (only returns 'complete' entries with pdfs)
    discovered = find_papers(queries, max_results=max_results)

    for paper in discovered:
        # Double-check completeness before persisting
        if not is_complete(paper):
            continue

        # Safely coerce year to int if possible
        year_val = None
        try:
            if paper.get("year") is not None:
                year_val = int(str(paper.get("year")).strip())
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
        except Exception as e:
            print(f"[DB Error] Failed to add paper '{paper.get('title')}': {e}")
            continue

        # augment paper dict with DB id and pdf_path (for caller)
        paper_out = dict(paper)
        paper_out["db_id"] = db_paper.id
        paper_out["pdf_path"] = paper.get("pdf_path")

        # Index into Qdrant with DB linkage (so retrieval results can reference the DB row)
        try:
            idx_payload = dict(paper_out)
            idx_payload["paper_id"] = db_paper.id
            # index_paper expects keys like 'pdf_path', 'title', etc.
            index_paper(idx_payload)
        except Exception as e:
            print(f"[Index Error] Failed to index paper id={db_paper.id} title='{paper.get('title')}': {e}")

        persisted.append(paper_out)

    return persisted
