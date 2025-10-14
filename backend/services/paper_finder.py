# backend/services/paper_finder.py

import os
import re
import requests
from typing import List, Dict, Optional

from services.rag_retriever import index_paper  # ✅ auto-index PDFs into Qdrant

# === Config ===
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

# ✅ Unpaywall requires a registered email (set in .env)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

# ✅ Central storage for PDFs
PAPER_STORAGE = os.path.join("storage", "papers")
os.makedirs(PAPER_STORAGE, exist_ok=True)


# === Utilities ===
def clean_filename(name: str) -> str:
    """Make a safe filename for saving PDFs."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)[:200]


def is_complete(paper: Dict) -> bool:
    """
    Check whether the paper has all required fields for an IEEE-style reference.
    Required fields: title, authors, year, venue (container-title), doi
    """
    required = ["title", "authors", "year", "venue", "doi"]
    for k in required:
        val = paper.get(k)
        if val is None:
            return False
        # treat empty strings as missing
        if isinstance(val, str) and not val.strip():
            return False
    # year should be an int-like (or string containing digits)
    year = paper.get("year")
    try:
        # allow str '2021' or int 2021
        if isinstance(year, str):
            if not re.search(r"\d{4}", year):
                return False
        else:
            int(year)
    except Exception:
        return False
    return True


# === CrossRef Search ===
def search_crossref(query: str, rows: int = 5) -> List[Dict]:
    """Search CrossRef for papers matching query and normalize metadata."""
    try:
        resp = requests.get(
            CROSSREF_API,
            params={"query": query, "rows": rows},
            timeout=10,
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
    except Exception as e:
        print(f"[CrossRef Error] {e}")
    return []


# === ArXiv Search ===
def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """Search arXiv API for papers. Note: many arXiv results lack DOI/venue; they will be filtered out."""
    import feedparser

    url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            # arXiv often doesn't have DOI or venue; include raw fields, completeness check will filter.
            title = getattr(entry, "title", "")
            authors = ", ".join(a.name for a in getattr(entry, "authors", [])) if getattr(entry, "authors", None) else None
            year = getattr(entry, "published", "").split("-")[0] if getattr(entry, "published", "") else None
            doi = entry.get("arxiv_doi") or None
            pdf_url = next(
                (l.href for l in entry.links if getattr(l, "type", "") == "application/pdf"), None
            )
            url = entry.get("link")
            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": "arXiv",  # explicit but note: if DOI missing this will be filtered
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
            timeout=10,
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
        resp = requests.get(url, stream=True, timeout=20)
        if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
            filename = clean_filename(title) + ".pdf"
            filepath = os.path.join(PAPER_STORAGE, filename)
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return filepath
    except Exception as e:
        print(f"[Download Error] {e}")
    return None


# === Main Entry Point ===
def find_papers(queries: List[str], max_results: int = 5) -> List[Dict]:
    """
    Given search queries, return metadata + local PDF paths (if OA available).
    Auto-indexes PDFs into Qdrant only for papers that contain all required fields
    for IEEE-style referencing (title, authors, year, venue, doi).
    """
    papers: List[Dict] = []

    for query in queries:
        # CrossRef
        crossref_results = search_crossref(query, rows=max_results)
        for paper in crossref_results:
            doi = paper.get("doi")
            # check Unpaywall for OA PDF only if DOI exists
            oa_link = get_unpaywall_oa_link(doi) if doi else None
            if oa_link:
                pdf_path = download_pdf(oa_link, paper["title"])
                if pdf_path:
                    paper["pdf_path"] = pdf_path
            # Only keep and index if paper has all required fields
            if is_complete(paper):
                # index only when pdf exists and paper is complete
                if paper.get("pdf_path"):
                    try:
                        index_paper(paper)
                    except Exception as e:
                        print(f"[Index Error] {e}")
                papers.append(paper)
            else:
                # skip incomplete entries
                continue

        # arXiv
        arxiv_results = search_arxiv(query, max_results=max_results)
        for paper in arxiv_results:
            pdf_url = paper.get("pdf_url")
            if pdf_url:
                pdf_path = download_pdf(pdf_url, paper["title"])
                if pdf_path:
                    paper["pdf_path"] = pdf_path
            # Only keep if it has all required fields (arXiv often lacks DOI -> will be skipped)
            if is_complete(paper):
                if paper.get("pdf_path"):
                    try:
                        index_paper(paper)
                    except Exception as e:
                        print(f"[Index Error] {e}")
                papers.append(paper)
            else:
                continue

    # Deduplicate by DOI or title (only complete papers are included)
    seen = set()
    unique_papers = []
    for p in papers:
        key = p.get("doi") or p.get("title")
        if key not in seen:
            seen.add(key)
            unique_papers.append(p)

    return unique_papers
