# backend/services/citation_manager.py
"""
Simple CrossRef lookup for a query term.
This returns the first reasonable hit or None.
"""
import requests

CROSSREF_API = "https://api.crossref.org/works"

def find_top_crossref_for_term(term):
    if not term:
        return None
    params = {"query": term, "rows": 3}
    try:
        r = requests.get(CROSSREF_API, params=params, timeout=6)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None
        top = items[0]
        return {
            "title": top.get("title", [""])[0],
            "doi": top.get("DOI"),
            "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in top.get("author", [])][:4],
            "year": top.get("issued", {}).get("date-parts", [[None]])[0][0]
        }
    except Exception:
        return None
