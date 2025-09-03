import re
import requests
import time
from typing import Dict

CROSSREF_API = "https://api.crossref.org/works"
MAX_RETRIES = 3  # retry limit
INITIAL_BACKOFF = 2  # seconds


def fetch_crossref_metadata(query: str) -> dict:
    """
    Fetch metadata from CrossRef using title or DOI.
    Retries on failure with exponential backoff.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                CROSSREF_API,
                params={"query": query, "rows": 1},
                timeout=10  # Increased timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("message", {}).get("items"):
                    return data["message"]["items"][0]
            else:
                print(f"[CrossRef Warning] Status {response.status_code} for query: {query}")
        except requests.exceptions.Timeout:
            print(f"[CrossRef Timeout] Attempt {attempt} for query: {query}")
        except Exception as e:
            print(f"[CrossRef Error] Attempt {attempt} for query '{query}': {e}")

        if attempt < MAX_RETRIES:
            backoff = INITIAL_BACKOFF * attempt
            print(f"[Retrying in {backoff}s]")
            time.sleep(backoff)

    return {}  # fallback if all retries fail


def format_reference(meta: dict) -> str:
    """
    Format metadata into IEEE-style reference.
    Example:
        [1] J. Smith, K. Lee, "Deep Learning for Computer Vision,"
        IEEE Transactions on Neural Networks, vol. 32, no. 5, pp. 120-130, 2021. DOI: 10.xxxx/xxxxx
    """
    title = meta.get("title", ["Unknown Title"])[0]
    authors_list = meta.get("author", [])
    authors = ", ".join(
        [f"{a.get('given','')[0]}. {a.get('family','')}" for a in authors_list]
    ) if authors_list else "Unknown Author"
    year = meta.get("issued", {}).get("date-parts", [[None]])[0][0] or "n.d."
    container = meta.get("container-title", [""])[0]  # Journal/Conference
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


def enrich_references(sections: Dict[str, str]) -> Dict[str, str]:
    """
    - Find [CITATION: ...] placeholders in text
    - Replace with [n]
    - Generate IEEE-style reference list using CrossRef metadata
    - Retry on API failure
    - Fallback to original query if all retries fail
    """
    citation_pattern = re.compile(r"\[CITATION:\s*([^\]]+)\]")
    unique_queries = []
    query_to_index = {}

    # ✅ Collect unique citation queries
    for text in sections.values():
        for match in citation_pattern.findall(text):
            query = match.strip()
            if query not in query_to_index:
                unique_queries.append(query)
                query_to_index[query] = len(unique_queries)  # 1-based index

    # ✅ Replace placeholders with [n]
    for sec_name, text in sections.items():
        sections[sec_name] = citation_pattern.sub(
            lambda m: f"[{query_to_index.get(m.group(1).strip(), '?')}]",
            text
        )

    # ✅ Build References list
    enriched_refs = []
    for query in unique_queries:
        meta = fetch_crossref_metadata(query)
        if meta:
            ref_text = format_reference(meta)
        else:
            ref_text = query  # fallback if API fails after retries
        enriched_refs.append(f"[{query_to_index[query]}] {ref_text}")

    sections["references"] = "\n".join(enriched_refs) if enriched_refs else "No references available."
    return sections
