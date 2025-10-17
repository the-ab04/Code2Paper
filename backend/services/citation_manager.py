import re
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from db import crud, schemas

# regex to detect citation placeholders like [CITATION: ...]
CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


def _paper_record_is_complete(paper: Any) -> bool:
    """Ensure a paper record has all required IEEE-style fields."""
    if not paper:
        return False
    required_fields = [
        getattr(paper, "title", None),
        getattr(paper, "authors", None),
        getattr(paper, "year", None),
        getattr(paper, "venue", None),
        getattr(paper, "doi", None),
    ]
    if any(not (v and str(v).strip()) for v in required_fields):
        return False
    return True


def _format_reference(paper: Any, index: int) -> str:
    """Format a paper record into IEEE-style reference text."""
    authors = paper.authors or "Unknown Author"
    title = paper.title or "Untitled"
    venue = paper.venue or ""
    year = getattr(paper, "year", "") or "n.d."
    doi = paper.doi or "N/A"
    return f"[{index}] {authors}, \"{title},\" {venue}, {year}. DOI: {doi}"


def enrich_references(sections: Dict[str, str], run_id: int, db: Session) -> Dict[str, str]:
    """
    Updated enrich_references():
      ✅ Works fully offline — only uses locally persisted & complete papers in DB.
      ✅ Resolves placeholders [CITATION: query] to local paper entries by DOI or title match.
      ✅ Builds the References section purely from DB data.
    """
    # Step 1: collect unique placeholders
    unique_queries: List[str] = []
    for text in sections.values():
        if not text:
            continue
        for match in CITATION_PATTERN.findall(text):
            q = match.strip()
            if q not in unique_queries:
                unique_queries.append(q)

    # Step 2: resolve placeholders to local DB papers only
    query_to_paper = {}
    unresolved = []
    for q in unique_queries:
        paper = None
        # try DOI exact match
        paper = db.query(crud.models.Paper).filter(crud.models.Paper.doi == q).first()
        if not paper:
            # try title partial match (case-insensitive)
            paper = (
                db.query(crud.models.Paper)
                .filter(crud.models.Paper.title.ilike(f"%{q}%"))
                .first()
            )

        if paper and _paper_record_is_complete(paper):
            query_to_paper[q] = paper
            try:
                crud.add_citation(db, run_id, paper.id, schemas.CitationBase(context=q, index=0))
            except Exception as e:
                print(f"[DB Error] failed to add citation for '{q}': {e}")
        else:
            unresolved.append(q)

    # Step 3: assign stable numeric indices
    for idx, (q, paper) in enumerate(query_to_paper.items(), start=1):
        try:
            db_cite = (
                db.query(crud.models.Citation)
                .filter(crud.models.Citation.run_id == run_id)
                .filter(crud.models.Citation.paper_id == paper.id)
                .order_by(crud.models.Citation.id.desc())
                .first()
            )
            if db_cite:
                db_cite.index = idx
                db.commit()
        except Exception as e:
            print(f"[DB Error] citation index update failed for '{q}': {e}")

    # Step 4: replace placeholders with numeric references
    def _replace_placeholder(match):
        q = match.group(1).strip()
        if q in query_to_paper:
            # find index from citation
            cite = (
                db.query(crud.models.Citation)
                .filter(crud.models.Citation.run_id == run_id)
                .filter(crud.models.Citation.paper_id == query_to_paper[q].id)
                .first()
            )
            if cite and cite.index:
                return f"[{cite.index}]"
        return match.group(0)

    for name, text in sections.items():
        if not text:
            continue
        sections[name] = CITATION_PATTERN.sub(_replace_placeholder, text)

    # Step 5: Build final References section purely from DB (only complete papers)
    citations = (
        db.query(crud.models.Citation)
        .filter(crud.models.Citation.run_id == run_id)
        .order_by(crud.models.Citation.index)
        .all()
    )

    formatted_refs = []
    for cite in citations:
        p = db.query(crud.models.Paper).filter(crud.models.Paper.id == cite.paper_id).first()
        if p and _paper_record_is_complete(p):
            formatted_refs.append(_format_reference(p, cite.index or len(formatted_refs) + 1))

    if formatted_refs:
        sections["references"] = "\n".join(formatted_refs)
    else:
        sections["references"] = "No references available."

    # Optional debug info
    sections["_unresolved_placeholders"] = unresolved
    return sections
