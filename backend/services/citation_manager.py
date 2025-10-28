# backend/services/citation_manager.py

import re
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from db import crud, schemas
import requests

CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)
CROSSREF_API = "https://api.crossref.org/works"
DEFAULT_HEADERS = {"User-Agent": "Code2Paper/1.0 (mailto:your-real-email@example.com)"}


def _safe_str(val: Any) -> str:
    """Return a trimmed string for a value or empty string if None."""
    if val is None:
        return ""
    return str(val).strip()


def _normalize_doi_like(token: str) -> str:
    """Return normalized DOI-like string for comparison (lowercased, strip common prefixes)."""
    if not token:
        return ""
    tok = token.strip()
    tok = re.sub(r"^https?://(dx\.)?doi\.org/", "", tok, flags=re.I)
    tok = re.sub(r"^doi:\s*", "", tok, flags=re.I)
    tok = re.sub(r"\s+", "", tok)
    return tok.lower()


def _crossref_lookup(query: str) -> Optional[Dict[str, Any]]:
    """
    Quick CrossRef lookup for a query. Returns the first matching item's normalized metadata
    (title, authors, year, doi, url, container-title) if found, else None.
    This is a lightweight fallback used when no local DB match exists.
    """
    if not query or not str(query).strip():
        return None
    try:
        resp = requests.get(
            CROSSREF_API,
            params={"query": query, "rows": 1},
            headers=DEFAULT_HEADERS,
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        msg = resp.json().get("message", {})
        items = msg.get("items", []) or []
        if not items:
            return None
        item = items[0]
        title = item.get("title", [""])[0] if item.get("title") else ""
        authors_list = item.get("author", []) or []
        authors = ", ".join(
            f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
            for a in authors_list
            if a.get("family") or a.get("given")
        ) or None
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        container = item.get("container-title", [""])[0] if item.get("container-title") else ""
        doi = item.get("DOI")
        url = item.get("URL")
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "venue": container,
            "doi": doi,
            "url": url,
            "source": "crossref",
        }
    except Exception:
        return None


def _find_paper(db: Session, query: str, run_id: Optional[int] = None) -> Optional[Any]:
    """
    Try to find a local Paper by:
      1) DOI exact match (normalized)
      2) title partial match (case-insensitive, tokenized)
      3) CrossRef fallback: if CrossRef yields a DOI/title, try to find it in DB;
         if not present and `run_id` provided, persist a new Paper row using CrossRef metadata
         (so downstream citation resolution can proceed).

    Returns Paper ORM object or None.
    """
    if not query:
        return None
    q = query.strip()

    # 1) Try DOI exact match using normalization
    doi_norm = _normalize_doi_like(q)
    try:
        if doi_norm:
            # prefer exact match (case-insensitive)
            try:
                paper = (
                    db.query(crud.models.Paper)
                    .filter(crud.models.Paper.doi.isnot(None))
                    .filter(crud.models.Paper.doi.ilike(f"%{doi_norm}%"))
                    .first()
                )
            except Exception:
                paper = db.query(crud.models.Paper).filter(crud.models.Paper.doi == q).first()
            if paper:
                return paper
    except Exception:
        # continue to next strategy on any DB error
        pass

    # 2) Try title partial match (case-insensitive)
    try:
        # Use simple containment match first
        paper = db.query(crud.models.Paper).filter(crud.models.Paper.title.ilike(f"%{q}%")).first()
        if paper:
            return paper

        # If direct containment fails, try tokenized matching (split into words and require at least one)
        tokens = [t.strip() for t in re.split(r"\W+", q) if t.strip()]
        if tokens:
            # build a simple OR-based ilike filter using SQL text (fallback) — keep simple and safe
            for tok in tokens[:6]:  # limit tokens to reduce query complexity
                paper = db.query(crud.models.Paper).filter(crud.models.Paper.title.ilike(f"%{tok}%")).first()
                if paper:
                    return paper
    except Exception:
        pass

    # 3) CrossRef fallback: try to fetch metadata and match/persist
    try:
        cr = _crossref_lookup(q)
        if cr and cr.get("doi"):
            doi_cr = _normalize_doi_like(cr.get("doi"))
            # try to find by DOI again (stricter)
            try:
                paper = (
                    db.query(crud.models.Paper)
                    .filter(crud.models.Paper.doi.isnot(None))
                    .filter(crud.models.Paper.doi.ilike(f"%{doi_cr}%"))
                    .first()
                )
                if paper:
                    return paper
            except Exception:
                pass

            # if not found, optionally create a new Paper record if run_id provided
            if run_id:
                try:
                    paper_base = schemas.PaperBase(
                        title=cr.get("title") or "Untitled",
                        authors=cr.get("authors"),
                        year=int(cr.get("year")) if cr.get("year") else None,
                        venue=cr.get("venue"),
                        doi=cr.get("doi"),
                        url=cr.get("url"),
                        pdf_path=None,
                    )
                    new_paper = crud.add_paper(db, run_id, paper_base)
                    return db.query(crud.models.Paper).filter(crud.models.Paper.id == new_paper.id).first()
                except Exception:
                    # if persisting fails, just continue
                    pass
        else:
            # If CrossRef returned no DOI but a title, try to match by returned title
            if cr and cr.get("title"):
                try:
                    paper = db.query(crud.models.Paper).filter(crud.models.Paper.title.ilike(f"%{cr.get('title')}%")).first()
                    if paper:
                        return paper
                except Exception:
                    pass
    except Exception:
        pass

    # No match found
    return None


def _ensure_citation_row(db: Session, run_id: int, paper_id: int, context: str) -> Any:
    """
    Ensure a Citation row exists linking the run and the paper.
    Returns the db citation object or None on failure.
    """
    if not run_id or not paper_id:
        return None
    try:
        # try to find existing citation for this run & paper
        c = (
            db.query(crud.models.Citation)
            .filter(crud.models.Citation.run_id == run_id)
            .filter(crud.models.Citation.paper_id == paper_id)
            .first()
        )
        if c:
            return c
        # create one
        new_cite = crud.add_citation(db, run_id, paper_id, schemas.CitationBase(context=context, index=0))
        return new_cite
    except Exception as e:
        # fallback attempt
        try:
            new_cite = crud.add_citation(db, run_id, paper_id, schemas.CitationBase(context=context, index=0))
            return new_cite
        except Exception:
            print(f"[Citation] Failed to ensure citation row for paper_id={paper_id}: {e}")
            return None


def _format_reference_any(paper: Any, index: int) -> str:
    """
    Format a reference line including any available fields.
    """
    if not paper:
        return f"[{index}] (Unknown reference)"

    parts: List[str] = []

    authors = _safe_str(getattr(paper, "authors", None))
    title = _safe_str(getattr(paper, "title", None))
    venue = _safe_str(getattr(paper, "venue", None))
    year = _safe_str(getattr(paper, "year", None))
    doi = _safe_str(getattr(paper, "doi", None))
    url = _safe_str(getattr(paper, "url", None))

    if authors:
        parts.append(authors)

    if title:
        parts.append(f"\"{title},\"")

    if venue:
        parts.append(venue)
    if year:
        parts.append(year)

    main = ", ".join(parts).strip()
    if not main:
        if doi:
            main = f"DOI: {doi}"
        elif title:
            main = f"\"{title},\""
        elif url:
            main = url
        else:
            main = "Unknown reference"

    if doi:
        if "doi" not in main.lower():
            return f"[{index}] {main}. DOI: {doi}"
        else:
            return f"[{index}] {main}"
    else:
        return f"[{index}] {main}"


def enrich_references(sections: Dict[str, str], run_id: int, db: Session) -> Dict[str, str]:
    """
    Resolve [CITATION: query] placeholders using local DB papers, with CrossRef fallback when helpful.
    Replaces placeholders with numeric references for resolved items, persists citation rows when needed,
    and builds a deterministic References section (ordered by assigned index).
    """
    # 1) collect unique placeholders in order
    unique_queries: List[str] = []
    for txt in sections.values():
        if not txt:
            continue
        for m in CITATION_PATTERN.findall(txt):
            q = m.strip()
            if q and q not in unique_queries:
                unique_queries.append(q)

    # 2) resolve queries to DB papers (local only, but allow CrossRef fallback and patient persistence)
    query_to_paper: Dict[str, Any] = {}
    unresolved: List[str] = []
    for q in unique_queries:
        # pass run_id so _find_paper may persist a CrossRef-discovered paper
        paper = _find_paper(db, q, run_id=run_id)
        if paper:
            query_to_paper[q] = paper
            _ensure_citation_row(db, run_id, paper.id, q)
        else:
            unresolved.append(q)

    # 3) fetch all citation rows for this run (we will set indices deterministically)
    try:
        db_citations = db.query(crud.models.Citation).filter(crud.models.Citation.run_id == run_id).all()
    except Exception:
        db_citations = []

    # Map paper_id -> Citation ORM (if exists)
    paperid_to_cite = {}
    for c in db_citations:
        if getattr(c, "paper_id", None):
            paperid_to_cite[c.paper_id] = c

    # 4) assign indices to resolved papers in the order of unique_queries (deterministic)
    assigned_index_map: Dict[int, int] = {}
    next_idx = 1
    for q in unique_queries:
        p = query_to_paper.get(q)
        if not p:
            continue
        # prefer existing citation row if present
        cite = paperid_to_cite.get(p.id)
        if cite and getattr(cite, "index", None) and cite.index > 0:
            assigned_index_map[p.id] = cite.index
            # ensure next_idx is beyond
            if cite.index >= next_idx:
                next_idx = cite.index + 1
            continue
        # else assign current next_idx and update/create citation row
        assigned_index_map[p.id] = next_idx
        try:
            if cite:
                cite.index = next_idx
            else:
                # create a new citation if none exists
                newc = crud.add_citation(db, run_id, p.id, schemas.CitationBase(context=q, index=next_idx))
                # reflect in mapping
                paperid_to_cite[p.id] = newc
            next_idx += 1
        except Exception:
            # if DB insert/update fails, still increment index and continue
            next_idx += 1
            continue

    # attempt a single commit after batch updates (best-effort)
    try:
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    # 5) Replace placeholders in section texts for resolved queries
    def _replace_match(m):
        inner = m.group(1).strip()
        p = query_to_paper.get(inner)
        if not p:
            # unresolved: keep original placeholder
            return m.group(0)
        # lookup assigned index
        idx = assigned_index_map.get(p.id)
        if idx:
            return f"[{idx}]"
        return m.group(0)

    for name, txt in list(sections.items()):
        if not txt:
            continue
        try:
            sections[name] = CITATION_PATTERN.sub(_replace_match, txt)
        except Exception as e:
            print(f"[Citation Replace] failed for section '{name}': {e}")

    # 6) Build References list from assigned indices and any remaining citation rows
    refs_by_index: Dict[int, str] = {}

    # Add assigned map entries first (deterministic order)
    for paper_id, idx in assigned_index_map.items():
        try:
            p = db.query(crud.models.Paper).filter(crud.models.Paper.id == paper_id).first()
            if p:
                refs_by_index[idx] = _format_reference_any(p, idx)
        except Exception:
            pass

    # Add any other citation rows (that might exist) ensuring no duplicate indices
    try:
        db_citations = db.query(crud.models.Citation).filter(crud.models.Citation.run_id == run_id).all()
    except Exception:
        db_citations = []

    for c in db_citations:
        idx = getattr(c, "index", None)
        if not idx or idx in refs_by_index:
            continue
        try:
            p = db.query(crud.models.Paper).filter(crud.models.Paper.id == c.paper_id).first()
            if p:
                refs_by_index[idx] = _format_reference_any(p, idx)
        except Exception:
            pass

    # Generate ordered reference lines
    refs_lines = []
    for idx in sorted(refs_by_index.keys()):
        refs_lines.append(refs_by_index[idx])

    if refs_lines:
        sections["references"] = "\n".join(refs_lines)
    else:
        # If nothing resolved, do not override an existing references section if present
        if not sections.get("references"):
            sections["references"] = "No references available."

    # attach debug info on unresolved placeholders
    sections["_unresolved_placeholders"] = unresolved

    # -------------------------------------------------------------
    # FINAL NORMALIZATION: convert figure placeholders to "Figure N"
    # (e.g., "Figure[1]" -> "Figure 1", "fig1" -> "Figure 1", "Fig. (1):" -> "Figure 1")
    # Also transforms grouped forms like "Figures [1] and [2]" -> "Figures 1 and 2"
    # Only targets tokens containing 'fig'/'figure' to avoid changing unrelated bracketed numbers.
    # -------------------------------------------------------------
    def _normalize_figure_placeholders_in_text(text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return text
        t = text

        # --- Handle grouped "Figures [...] [..] and [...]" forms first ---
        # Matches: "Figures [1] and [2]", "Figures: [1, 2]", "Figures [1][2]" etc.
        group_pat = re.compile(
            r'\b(Figures?|Figures?:?)\s*[:\.\-]?\s*((?:[\(\[]\s*\d{1,4}\s*[\)\]]\s*(?:,|\s+and\s+|\s*[-–—]\s*)?)+)',
            flags=re.I,
        )

        def _group_repl(m):
            head = m.group(1) or "Figures"
            body = m.group(2) or ""
            # extract all numeric tokens and ranges if present
            # preserve ranges like "1-3" by capturing them first
            ranges = re.findall(r'(\d{1,4}\s*[-–—]\s*\d{1,4})', body)
            # extract individual numbers
            nums = re.findall(r'(\d{1,4})', body)
            parts: List[str] = []

            # If ranges found, include them (they'll also produce numbers in nums; keep ranges first)
            used_numbers = set()
            for r in ranges:
                # normalize dashes
                r_norm = re.sub(r'\s*[-–—]\s*', '-', r)
                parts.append(r_norm)
                # mark numbers in range as used
                mnums = re.findall(r'(\d{1,4})', r)
                for mn in mnums:
                    used_numbers.add(mn)

            # Add remaining single numbers in appearance order
            for n in nums:
                if n in used_numbers:
                    continue
                parts.append(n)
                used_numbers.add(n)

            # Build connective string: "1", "1 and 2", "1, 2 and 3"
            if not parts:
                return head  # fallback: return only the head word
            if len(parts) == 1:
                # singular -> use "Figure N" (singular)
                return f"Figure {parts[0]}"
            else:
                # plural head normalization: ensure 'Figures' (plural)
                head_plural = "Figures"
                if len(parts) == 2:
                    return f"{head_plural} {parts[0]} and {parts[1]}"
                # more than 2: comma separated with 'and' before last
                return f"{head_plural} " + ", ".join(parts[:-1]) + f" and {parts[-1]}"

        t = group_pat.sub(_group_repl, t)

        # Pattern 1: fig/figure with optional punctuation and optional brackets/parentheses around the number
        pat = re.compile(
            r'\b(?:fig(?:\.|ure)?|figure)\s*[:\.\-]?\s*[\(\[\s]*\s*(\d{1,4})\s*[\)\]\s]*',
            flags=re.I,
        )

        t = pat.sub(lambda m: f"Figure {m.group(1)}", t)

        # Pattern 2: tightly joined forms like "fig1" or "figure1"
        pat2 = re.compile(r'\b(?:fig(?:ure)?)(\d{1,4})\b', flags=re.I)
        t = pat2.sub(lambda m: f"Figure {m.group(1)}", t)

        # Collapse multiple spaces introduced by replacements
        t = re.sub(r'\s{2,}', ' ', t).strip()
        return t

    try:
        # Apply normalization to all textual sections except 'references' and internal keys starting with '_'
        for k in list(sections.keys()):
            if not isinstance(sections.get(k), str):
                continue
            if k == "references" or k.startswith("_"):
                continue
            try:
                sections[k] = _normalize_figure_placeholders_in_text(sections[k])
            except Exception:
                # do not break the flow if normalization fails for a section
                pass
    except Exception:
        # swallow errors in final normalization
        pass

    return sections
