# backend/services/llm_generator.py

from typing import Dict, Any, List, Optional
from config import GROQ_API_KEY, MODEL_NAME
from langchain_groq import ChatGroq
from services.rag_retriever import query_chunks
import json
import re

# === System-level guidance for all prompts ===
SYSTEM_PROMPT = (
    "You are a highly capable assistant that writes detailed, research-style academic paper sections."
    " Be factual, concise, and formal. When asked to produce a section, produce only the section text"
    " (no YAML/frontmatter). For the References step, return valid JSON (an array of strings)."
    " When inserting citation placeholders use the format [CITATION: query]. Prefer using a DOI as the"
    " query WHENEVER a DOI is available; otherwise use a clear paper title. Do not invent DOIs."
)

# === Per-section instructions (tailored prompts) ===
SECTION_INSTRUCTIONS = {
    "title": (
        "Write a clear, descriptive paper title (8–16 words) summarizing the main contribution. "
        "Avoid citations and punctuation-heavy qualifiers. Keep it formal and concise."
    ),
    "abstract": (
        "Write a single-paragraph abstract (120–220 words) summarizing: problem, approach, key results, "
        "and main contributions. Do NOT include citation placeholders in the abstract; omit them if present."
    ),
    "introduction": (
        "Write an Introduction: motivate the problem, give short background, state the gap, and list the "
        "contributions of this work in bullet-like sentences (but keep the output as a coherent paragraph(s)). "
        "Insert citation placeholders as [CITATION: query] when referencing prior work. Prefer DOIs if known."
    ),
    "methods": (
        "Write a Methods section describing the model/architecture, datasets, training, hyperparameters, and any "
        "implementation details necessary to reproduce the work. Use citation placeholders [CITATION: query] where "
        "appropriate to reference methods from the literature. Prefer DOIs if known."
    ),
    "experiments": (
        "Write an Experiments section describing the evaluation protocol, datasets/splits, baselines, metrics, and "
        "experimental setup. Use citation placeholders where comparing to prior works. Prefer DOIs if known."
    ),
    "results": (
        "Write a Results section summarizing quantitative and qualitative findings, include key tables/metrics "
        "phrases (no raw tables), and mention notable observations. Use citation placeholders where referencing "
        "benchmarks or prior reported numbers. Prefer DOIs if known."
    ),
    "discussion": (
        "Write a Discussion: interpret results, discuss strengths and weaknesses, possible causes of behavior, "
        "and practical implications. Use citation placeholders as needed. Prefer DOIs if known."
    ),
    "conclusion": (
        "Write a short Conclusion (2–4 sentences) summarizing the work and listing future directions."
    ),
    # 'references' is handled separately and must return valid JSON array
    "references": (
        "Return a JSON array of citation placeholders (strings). Each item should be a short query suitable for "
        "CrossRef lookup or the format used by downstream citation resolver, e.g. a paper title or DOI. "
        "Prefer DOIs if available. Return only JSON array, nothing else."
    )
}

# patterns
CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


# === Helpers: LLM client / safe invoke ===
def _get_llm(model_name: Optional[str] = None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in config.")
    return ChatGroq(model=model_name or MODEL_NAME, temperature=0.15, groq_api_key=GROQ_API_KEY)


def _safe_invoke(llm, prompt: str) -> str:
    """
    Call the LLM and return assistant content as str (robust).
    Raises RuntimeError with a helpful message when the underlying client fails.
    """
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", str(resp))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as e:
        # Provide a clear error for upstream handling (e.g., rate limit)
        raise RuntimeError(f"LLM invocation failed: {e}")


# tolerant JSON array extraction (handles code fences, trailing commas, plain quoted tokens, or bare [CITATION: ...])
def _extract_json_array(text: str) -> List[str]:
    """
    Extract a JSON array from a string and return Python list.
    Accepts arrays like ["a","b"] possibly surrounded by text or code fences.
    Falls back to extracting quoted strings or [CITATION: ...] tokens.
    """
    if not text:
        return []

    # strip common code fences
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)

    # try to find the first balanced array-like bracket pair
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        arr_text = text[start:end + 1]
        # attempt JSON parse directly
        try:
            parsed = json.loads(arr_text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except Exception:
            # try small repairs: remove trailing commas, newlines etc.
            repaired = re.sub(r",\s*([}\]])", r"\1", arr_text)
            repaired = repaired.replace("\n", " ")
            # extract double-quoted tokens first
            items = re.findall(r'"([^"]+)"', repaired)
            if items:
                return [i.strip() for i in items]
            # last resort: split on commas inside the bracket (very tolerant)
            inner = repaired.strip()[1:-1]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
            return parts

    # fallback: detect placeholders like [CITATION: ...]
    ci = CITATION_PATTERN.findall(text)
    if ci:
        return [c.strip() for c in ci]

    # fallback: any quoted tokens in the text
    items = re.findall(r'"([^"]+)"', text)
    return [i.strip() for i in items] if items else []


def _remove_citation_placeholders(text: str) -> str:
    """Remove any [CITATION: ...] placeholders from text."""
    return CITATION_PATTERN.sub("", text)


# small normalizers
def _normalize_doi(token: str) -> str:
    """Normalize a DOI-like token: strip common prefixes and lowercase."""
    if not token:
        return ""
    token = token.strip()
    # remove URL-like prefixes
    token = re.sub(r"^https?://(dx\.)?doi\.org/", "", token, flags=re.I)
    token = re.sub(r"^doi:\s*", "", token, flags=re.I)
    return token.strip().lower()


def _normalize_title_for_match(title: str) -> str:
    """Lowercase & collapse whitespace for title matching."""
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _format_candidate_block(candidate_papers: List[Dict[str, Any]]) -> str:
    """
    Build a human-readable candidate papers block that will be injected into prompts.
    Each line: [n] DOI: <doi> | Title: <title> | Authors: <authors> | Year: <year>
    If DOI missing, DOI: (none)
    """
    lines = []
    for idx, p in enumerate(candidate_papers, start=1):
        doi = p.get("doi") or ""
        title = (p.get("title") or "").strip()
        authors = (p.get("authors") or "").strip()
        year = p.get("year") or ""
        lines.append(
            f"[{idx}] DOI: {doi or '(none)'} | Title: {title or '(no title)'} | Authors: {authors or '(no authors)'} | Year: {year}"
        )
    block = "=== Candidate Papers (allowed to cite) ===\n" + "\n".join(lines) + "\n"
    block += (
        "Important: You MUST only cite papers from the numbered candidate list above. "
        "When creating a citation placeholder, prefer the DOI if present; otherwise use the title exactly as shown above.\n"
    )
    return block


# === Main public function ===
def generate_sections(
    facts: Dict[str, Any],
    sections_to_generate: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    use_rag: bool = True,
    top_k: int = 5,
    rag_context_override: Optional[str] = None,
    candidate_papers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Generate paper sections.

    Args:
      facts: notebook-extracted facts dict (markdown, code, logs, datasets, metrics, ...).
      sections_to_generate: optional list of section names to generate (e.g. ["abstract","methods"]).
                           If None, generate all sections.
      model_name: override model name for Groq (optional).
      use_rag: whether to include retrieved literature context.
      top_k: number of retrieved chunks to use.
      rag_context_override: if provided, this string will be inserted as the retrieved context
                            instead of performing a live query (useful for testing or custom context).
      candidate_papers: optional list of paper metadata dicts (title, doi, authors, year, venue, ...) that
                        the LLM is allowed to cite. If provided, the LLM is instructed to ONLY cite from this list.

    Returns:
      dict mapping section_name -> section_text. For 'references' the string contains newline-separated
      placeholders like "[CITATION: ...]". Only requested/generated sections are returned.
    """
    llm = _get_llm(model_name)

    # Allowed order and canonical section names
    section_order = [
        "title", "abstract", "introduction", "methods",
        "experiments", "results", "discussion", "conclusion", "references"
    ]

    # Normalize requested sections if provided; otherwise generate all
    if sections_to_generate is None:
        requested = section_order[:]
    else:
        requested = []
        lower_to_canon = {s: s for s in section_order}
        for s in sections_to_generate:
            if not s:
                continue
            key = str(s).strip().lower()
            if key in lower_to_canon:
                requested.append(lower_to_canon[key])
        if not requested:
            requested = section_order[:]

    # Prepare notebook context snippet
    md = "\n\n".join(facts.get("markdown", []))[:4000]
    code_hint = "\n".join(facts.get("code", [])[:3])
    logs = "\n".join(facts.get("logs", [])[:10])
    datasets = ", ".join(facts.get("datasets", []))
    metrics = ", ".join(facts.get("metrics", []))

    # RAG: retrieve relevant chunks and create a compact context block
    rag_context = ""
    if rag_context_override:
        rag_context = rag_context_override
    elif use_rag:
        queries = []
        if facts.get("task"):
            queries.append(facts["task"])
        queries.extend(facts.get("datasets", [])[:2])
        queries.extend(facts.get("methods", [])[:2] if facts.get("methods") else [])

        if queries:
            chunks = query_chunks(queries, top_k=top_k)
            if chunks:
                rag_context = "=== Retrieved Literature Chunks ===\n"
                for idx, ch in enumerate(chunks, start=1):
                    src = ch.get("doi") or ch.get("title") or f"Source {idx}"
                    snippet = ch.get("text") or ch.get("chunk") or ""
                    snippet_short = (snippet[:800] + "...") if len(snippet) > 800 else snippet
                    rag_context += f"[{idx}] {snippet_short} (Source: {src})\n"

    # Candidate papers block (explicitly allowed cite list)
    candidate_block = _format_candidate_block(candidate_papers) if candidate_papers else ""

    # Construct a common context header inserted into each prompt
    common_context = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{candidate_block}\n"
        f"Notebook facts (sample):\n{md}\n\n"
        f"Code sample:\n{code_hint}\n\n"
        f"Training logs:\n{logs}\n\n"
        f"Datasets: {datasets}\n"
        f"Metrics: {metrics}\n\n"
        f"{rag_context}\n"
    )

    produced: Dict[str, str] = {}

    # Precompute candidate lookups for filtering refs (if any)
    candidate_doi_map = {}
    candidate_title_map = []
    if candidate_papers:
        for p in candidate_papers:
            doi_norm = _normalize_doi(p.get("doi") or "")
            if doi_norm:
                candidate_doi_map[doi_norm] = p
            t_norm = _normalize_title_for_match(p.get("title") or "")
            if t_norm:
                candidate_title_map.append((t_norm, p))

    # iterate and produce each requested section independently following canonical order
    for section in section_order:
        if section not in requested:
            continue

        instr = SECTION_INSTRUCTIONS.get(section, "")

        if section == "references":
            # ask explicitly for JSON array
            prompt = (
                f"{common_context}\n"
                f"Now produce ONLY a JSON array (no other text) where each element is a short citation query "
                f"that can be used to look up the reference (title or DOI). Example: [\"Title A\", \"10.xxxx/abcd\"].\n\n"
                f"{instr}\n"
            )
            raw = _safe_invoke(llm, prompt)
            refs_list = _extract_json_array(raw)

            # If candidate_papers provided, accept only those matching candidate DOIs/titles
            final_refs: List[str] = []
            if candidate_papers and refs_list:
                for r in refs_list:
                    token = r.strip()
                    if not token:
                        continue
                    # remove surrounding [CITATION: ...] if present
                    m = re.match(r"^\[CITATION:\s*(.+?)\]$", token, flags=re.I)
                    if m:
                        token = m.group(1).strip()

                    doi_norm = _normalize_doi(token)
                    matched = None
                    if doi_norm and doi_norm in candidate_doi_map:
                        matched = candidate_doi_map[doi_norm]
                    else:
                        # try title partial match
                        token_norm = _normalize_title_for_match(token)
                        for t_norm, p in candidate_title_map:
                            if token_norm in t_norm or t_norm in token_norm:
                                matched = p
                                break
                    if matched:
                        # prefer DOI when available; else prefer canonical title from candidate record
                        use = (matched.get("doi") or matched.get("title") or token).strip()
                        final_refs.append(use)
                    else:
                        # skip any external refs not in candidate list
                        continue
            else:
                # accept whatever parsed
                final_refs = [r.strip() for r in refs_list if r and r.strip()]

            # wrap to canonical placeholders
            placeholder_lines = []
            for item in final_refs:
                if not item:
                    continue
                if re.match(r"^\[CITATION:", item, re.I):
                    placeholder_lines.append(item)
                else:
                    placeholder_lines.append(f"[CITATION: {item}]")

            produced["references"] = "\n".join(placeholder_lines)
        else:
            # textual section
            extra = ""
            if rag_context:
                extra = (
                    "Use the retrieved literature snippets above to ground claims. When referencing a specific paper, "
                    "prefer using the DOI shown in the 'Source:' parenthesis if present; otherwise use the paper title."
                )
            if candidate_block:
                extra = (extra + "\n" if extra else "") + (
                    "You MUST only cite papers from the 'Candidate Papers' list above. Use DOI if present; otherwise use the title exactly as shown."
                )

            prompt = (
                f"{common_context}\n"
                f"Write the {section.upper()} section.\n"
                f"{instr}\n\n"
                f"{extra}\n\n"
                "Return only the section text (no surrounding JSON or markdown fences)."
            )
            raw = _safe_invoke(llm, prompt)
            text = str(raw).strip()

            # remove fenced code block if present
            fence_match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, flags=re.S)
            if fence_match:
                text = fence_match.group(1).strip()

            # For abstract only: remove citation placeholders entirely
            if section == "abstract":
                text = _remove_citation_placeholders(text).strip()
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\s+([.,;:])", r"\1", text)

            produced[section] = text

    return produced
