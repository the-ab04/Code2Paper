# backend/services/citation_manager.py

import re
import requests
from typing import Dict, List
from sqlalchemy.orm import Session
from db import crud, schemas

CROSSREF_API = "https://api.crossref.org/works"


# === CrossRef Metadata Fetch ===
def fetch_crossref_metadata(query: str) -> dict:
    """
    Fetch metadata from CrossRef using a title or DOI query.
    Returns the first matching metadata item or an empty dict if not found.
    """
    try:
        response = requests.get(
            CROSSREF_API,
            params={"query": query, "rows": 1, "mailto": "your-real-email@example.com"},  # 👈 replace with real email
            timeout=8,
        )
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items:
                return items[0]
    except Exception as e:
        print(f"[CrossRef Error] Failed for '{query}': {e}")
    return {}


# === Format Reference (IEEE style) ===
def format_reference(meta: dict) -> str:
    """Convert CrossRef metadata into IEEE-style reference."""
    title = meta.get("title", ["Unknown Title"])[0]
    authors_list = meta.get("author", [])
    authors = ", ".join(
        [f"{a.get('given','')[:1]}. {a.get('family','')}" for a in authors_list if a.get("family")]
    ) if authors_list else "Unknown Author"
    year = meta.get("issued", {}).get("date-parts", [[None]])[0][0] or "n.d."
    container = meta.get("container-title", [""])[0]
    volume = meta.get("volume", "")
    issue = meta.get("issue", "")
    pages = meta.get("page", "")
    doi = meta.get("DOI", "N/A")

    ref = f"{authors}, \"{title},\" {container}"
    if volume:
        ref += f", vol. {volume}"
    if issue:
        ref += f", no. {issue}"
    if pages:
        ref += f", pp. {pages}"
    ref += f", {year}. DOI: {doi}"
    return ref


# === Main Function ===
def enrich_references(sections: Dict[str, str], run_id: int, db: Session) -> Dict[str, str]:
    """
    Process [CITATION: ...] placeholders:
      1. Detect unique queries
      2. Fetch metadata, insert into DB
      3. Replace placeholders with [n]
      4. Build numbered References section
    """
    citation_pattern = re.compile(r"\[CITATION:\s*([^\]]+)\]")

    unique_queries: List[str] = []
    query_to_index: Dict[str, int] = {}

    # Step 1: Collect unique queries
    for sec_text in sections.values():
        for match in citation_pattern.findall(sec_text):
            query = match.strip()
            if query not in query_to_index:
                unique_queries.append(query)
                query_to_index[query] = len(unique_queries)  # [1], [2], ...

    # Step 2: Replace placeholders in text
    for sec_name, sec_text in sections.items():
        sections[sec_name] = citation_pattern.sub(
            lambda m: f"[{query_to_index.get(m.group(1).strip(), '?')}]",
            sec_text,
        )

    # Step 3: Insert into DB + build reference list
    enriched_refs: List[str] = []
    for query in unique_queries:
        meta = fetch_crossref_metadata(query)

        if meta:
            paper_schema = schemas.PaperBase(
                title=meta.get("title", ["Unknown Title"])[0],
                authors=", ".join(
                    f"{a.get('given','')} {a.get('family','')}" for a in meta.get("author", [])
                ) if meta.get("author") else None,
                year=meta.get("issued", {}).get("date-parts", [[None]])[0][0],
                venue=meta.get("container-title", [""])[0],
                doi=meta.get("DOI"),
                url=meta.get("URL"),
            )
            # Save paper + citation in DB
            db_paper = crud.add_paper(db, run_id, paper_schema)
            crud.add_citation(
                db, run_id, db_paper.id,
                schemas.CitationBase(context=query, index=query_to_index[query])
            )

            ref_text = format_reference(meta)
        else:
            ref_text = query  # fallback to raw query

        enriched_refs.append(f"[{query_to_index[query]}] {ref_text}")

    # Step 4: Update References section
    sections["references"] = "\n".join(enriched_refs) if enriched_refs else "No references available."
    return sections
