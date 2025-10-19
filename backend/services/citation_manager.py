# backend/services/citation_manager.py

import re
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from db import crud, schemas

CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


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
    return tok.lower()


def _find_paper(db: Session, query: str) -> Optional[Any]:
    """
    Try to find a local Paper by:
      1) DOI exact match (normalized)
      2) title partial match (case-insensitive)
      3) URL or other fuzzy fallback (not implemented)
    Returns Paper ORM object or None.
    """
    if not query:
        return None
    q = query.strip()

    # Try DOI exact match using normalization
    doi_norm = _normalize_doi_like(q)
    if doi_norm:
        try:
            # compare normalized DOIs stored in DB (we assume stored DOI is raw; compare lowercased)
            paper = db.query(crud.models.Paper).filter(crud.models.Paper.doi.ilike(doi_norm)).first()
            if paper:
                return paper
        except Exception:
            # fallback: direct equality
            try:
                paper = db.query(crud.models.Paper).filter(crud.models.Paper.doi == q).first()
                if paper:
                    return paper
            except Exception:
                pass

    # Try title partial match (case-insensitive)
    try:
        paper = db.query(crud.models.Paper).filter(crud.models.Paper.title.ilike(f"%{q}%")).first()
        if paper:
            return paper
    except Exception:
        pass

    # No match
    return None


def _ensure_citation_row(db: Session, run_id: int, paper_id: int, context: str) -> Any:
    """
    Ensure a Citation row exists linking the run and the paper.
    Returns the db citation object.
    """
    try:
        # Try to find an existing citation row for same run & paper & context (or any)
        c = (
            db.query(crud.models.Citation)
            .filter(crud.models.Citation.run_id == run_id)
            .filter(crud.models.Citation.paper_id == paper_id)
            .first()
        )
        if c:
            return c
        # Create via crud helper if available (crud.add_citation should return the ORM instance)
        new_cite = crud.add_citation(db, run_id, paper_id, schemas.CitationBase(context=context, index=0))
        return new_cite
    except Exception:
        # fallback: attempt low-level insert via crud helper if it raises
        try:
            new_cite = crud.add_citation(db, run_id, paper_id, schemas.CitationBase(context=context, index=0))
            return new_cite
        except Exception as e:
            print(f"[Citation] Failed to ensure citation row for paper_id={paper_id}: {e}")
            return None


def _format_reference_any(paper: Any, index: int) -> str:
    """
    Format a reference line including any available fields.
    Example outputs (flexible):
      [1] J. Doe, "Title of Paper," Conference XYZ, 2021. DOI: 10.xxxx/abcd
      [2] Unknown Author, "Untitled," 2019. DOI: ...
      [3] J. Smith, "Some Title,"
    We include only fields that exist; missing bits are skipped but we always produce a line.
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

    # authors
    if authors:
        parts.append(authors)
    else:
        # do not force authors, but we can show 'Unknown Author' if nothing else
        pass

    # title (prefer to wrap in quotes)
    if title:
        parts.append(f"\"{title},\"")
    else:
        # If no title, maybe show DOI or url as short identifier later
        pass

    # venue/year
    if venue:
        parts.append(venue)
    if year:
        parts.append(year)

    # assemble main portion
    main = ", ".join(parts).strip()
    if not main:
        # fallback minimal identification: DOI or title or URL or 'Unknown'
        if doi:
            main = f"DOI: {doi}"
        elif title:
            main = f"\"{title},\""
        elif url:
            main = url
        else:
            main = "Unknown reference"

    # append DOI if present and not already included
    if doi:
        # avoid repeating DOI if main already contains DOI text
        if "doi" not in main.lower():
            return f"[{index}] {main}. DOI: {doi}"
        else:
            return f"[{index}] {main}"
    else:
        return f"[{index}] {main}"


def enrich_references(sections: Dict[str, str], run_id: int, db: Session) -> Dict[str, str]:
    """
    Resolve [CITATION: query] placeholders using only local DB papers.
    For any resolved paper we will:
      - create or reuse a Citation row (index will be assigned deterministically)
      - replace placeholders with numeric [n]
      - build 'references' using any available fields from the Paper row

    If a placeholder cannot be resolved to a local Paper, it remains unchanged and is listed in
    sections['_unresolved_placeholders'] for debugging.
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

    # 2) resolve queries to DB papers (local only)
    query_to_paper: Dict[str, Any] = {}
    unresolved: List[str] = []
    for q in unique_queries:
        paper = _find_paper(db, q)
        if paper:
            query_to_paper[q] = paper
            # ensure a citation row exists (index will be set later)
            _ensure_citation_row(db, run_id, paper.id, q)
        else:
            unresolved.append(q)

    # 3) assign stable numeric indices to citations (order reflects the order of resolved placeholders)
    # We'll fetch citations for this run and assign indices for those papers we resolved (in resolved order).
    try:
        # get all citations for this run (we will update indices for resolved ones)
        db_citations = db.query(crud.models.Citation).filter(crud.models.Citation.run_id == run_id).all()
    except Exception:
        db_citations = []

    # map paper_id -> citation ORM object (choose latest if multiple)
    paperid_to_cite = {}
    for c in db_citations:
        if getattr(c, "paper_id", None):
            paperid_to_cite[c.paper_id] = c

    # Now enumerate resolved papers in the same order as query_to_paper
    assigned_index_map: Dict[int, int] = {}
    next_idx = 1
    for q in query_to_paper.keys():
        p = query_to_paper[q]
        if not p:
            continue
        cite = paperid_to_cite.get(p.id)
        if cite:
            cite.index = next_idx
            try:
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            assigned_index_map[p.id] = next_idx
            next_idx += 1
        else:
            # create citation row (if for some reason _ensure_citation_row didn't produce it)
            try:
                new_c = crud.add_citation(db, run_id, p.id, schemas.CitationBase(context=q, index=next_idx))
                assigned_index_map[p.id] = next_idx
                next_idx += 1
            except Exception:
                # skip but continue numbering
                assigned_index_map[p.id] = next_idx
                next_idx += 1

    # 4) Replace placeholders in section texts for resolved queries
    def _replace_match(m):
        inner = m.group(1).strip()
        p = query_to_paper.get(inner)
        if not p:
            # unresolved: keep placeholder
            return m.group(0)
        # find citation index (from DB or assigned_index_map)
        c = None
        try:
            c = db.query(crud.models.Citation).filter(crud.models.Citation.run_id == run_id, crud.models.Citation.paper_id == p.id).first()
        except Exception:
            c = None
        idx = None
        if c and getattr(c, "index", None):
            idx = c.index
        else:
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

    # 5) Build References list from the assigned citation indices (ordered by index)
    refs_by_index: Dict[int, str] = {}
    try:
        db_citations = db.query(crud.models.Citation).filter(crud.models.Citation.run_id == run_id).all()
    except Exception:
        db_citations = []

    # ensure we include citations for resolved papers first (in assigned_index_map order)
    for paper_id, idx in assigned_index_map.items():
        try:
            p = db.query(crud.models.Paper).filter(crud.models.Paper.id == paper_id).first()
            if p:
                refs_by_index[idx] = _format_reference_any(p, idx)
        except Exception:
            pass

    # include any other citation rows that might exist (and haven't been added above)
    for c in db_citations:
        idx = getattr(c, "index", None) or None
        if not idx:
            # assign a sequential index if missing and not already in refs_by_index
            idx = max(refs_by_index.keys(), default=0) + 1
            try:
                c.index = idx
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
        if idx in refs_by_index:
            continue
        try:
            p = db.query(crud.models.Paper).filter(crud.models.Paper.id == c.paper_id).first()
            if p:
                refs_by_index[idx] = _format_reference_any(p, idx)
        except Exception:
            pass

    # create ordered list of references by numeric index
    refs_lines = []
    for idx in sorted(refs_by_index.keys()):
        refs_lines.append(refs_by_index[idx])

    if refs_lines:
        sections["references"] = "\n".join(refs_lines)
    else:
        sections["references"] = "No references available."

    # attach debug info on unresolved placeholders
    sections["_unresolved_placeholders"] = unresolved

    return sections
