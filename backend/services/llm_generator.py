# backend/services/llm_generator.py

from typing import Dict, Any, List, Optional
from config import GROQ_API_KEY, MODEL_NAME
# NOTE: langchain_groq and query_chunks are imported lazily inside functions to
# avoid circular imports and reduce cold-start time.
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
        "Avoid citations, dataset names in parentheses, dates, file names, UUIDs, or punctuation-heavy qualifiers. "
        "Keep it formal and concise."
    ),
    "abstract": (
        "Write a single-paragraph abstract (120–220 words) that summarizes: the problem, approach, key results, "
        "and main contributions. Do NOT include citation placeholders in the abstract; remove them if present."
    ),
    "introduction": (
        "Write an Introduction: motivate the problem, provide brief background, identify the gap, and list the "
        "contributions of this work. Use citation placeholders [CITATION: query] when referencing prior work. "
        "Prefer DOIs when available. Keep the tone formal and concise."
    ),
    "literature_review": (
        "Write a Literature Review that synthesizes prior work relevant to the task. Group related works by theme/approach "
        "(3–6 themes), briefly compare representative methods/benchmarks for each theme, and highlight remaining gaps "
        "that justify the present work. Insert citation placeholders [CITATION: query] when referencing specific papers "
        "(prefer DOIs). Aim for synthesis and comparison rather than a long unstructured list."
    ),
    "methods": (
        "Write a Methods section describing the model/architecture, datasets, preprocessing, training, hyperparameters, "
        "and implementation details necessary to reproduce results. Use citation placeholders [CITATION: query] where "
        "relevant to reference standard model components or datasets."
    ),
    "experiments": (
        "Write an Experiments section describing datasets/splits, evaluation protocol, baselines, metrics, and the exact "
        "experimental procedure (runs/seeds/hyperparameter sweep). Use citation placeholders only when comparing to prior work."
    ),
    "results": (
        "Write a Results section summarizing quantitative and qualitative findings. Include key metric phrases (e.g., "
        "\"achieved 92% top-1 accuracy\") and salient observations. Do NOT include citation placeholders; omit them if present."
    ),
    "conclusion": (
        "Write a short Conclusion (2–4 sentences) summarizing the main takeaways and proposing future directions. "
        "Do NOT include citation placeholders in the conclusion; remove them if present."
    ),
    # 'references' is handled separately and must return valid JSON array
    "references": (
        "Return a JSON array of citation placeholders (strings). Each element should be a short query suitable for "
        "CrossRef lookup or the format used by downstream citation resolver, e.g. a paper title or DOI. Prefer DOIs if available. "
        "Return only a JSON array, nothing else."
    )
}

# patterns
CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


# === Helpers: LLM client / safe invoke ===
def _get_llm(model_name: Optional[str] = None):
    """
    Lazily import and construct the LLM client. This avoids importing the 3rd-party
    client at module import time (prevents circular imports and speeds startup).
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in config.")
    try:
        # Lazy import to avoid heavy import at module load time
        from langchain_groq import ChatGroq
    except Exception as e:
        raise RuntimeError(f"Failed to import langchain_groq.ChatGroq: {e}")

    try:
        return ChatGroq(model=model_name or MODEL_NAME, temperature=0.15, groq_api_key=GROQ_API_KEY)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq client: {e}")


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


# === Facts summary helper ===
def _build_facts_summary(facts: Dict[str, Any]) -> str:
    """
    Build a short, high-signal summary of the notebook facts to include in prompts.
    Prefer compact bullet-like sentences that convey task, datasets, methods, frameworks and top metrics.
    """
    parts = []
    task = facts.get("task")
    if task:
        parts.append(f"Task: {task}")

    datasets = facts.get("datasets", []) or []
    if datasets:
        parts.append("Datasets: " + ", ".join(datasets[:4]))

    methods = facts.get("methods", []) or []
    if methods:
        parts.append("Methods: " + ", ".join(methods[:4]))

    frameworks = facts.get("frameworks", []) or []
    if frameworks:
        parts.append("Frameworks: " + ", ".join(frameworks[:4]))

    metrics = facts.get("metrics", []) or []
    if metrics:
        parts.append("Metrics: " + ", ".join(metrics[:4]))

    # include a short markdown preview (first meaningful line)
    md_blocks = facts.get("markdown", []) or []
    md_preview = ""
    for m in md_blocks:
        if isinstance(m, str) and m.strip():
            md_preview = m.strip().splitlines()[0]
            if len(md_preview) > 200:
                md_preview = md_preview[:200] + "..."
            break
    if md_preview:
        parts.append(f"Notebook note: {md_preview}")

    if not parts:
        return ""
    return " | ".join(parts)


# === Title sanitization helper ===
def _sanitize_title(raw_title: str, fallback: str = "") -> str:
    """
    Clean up a title produced by the LLM:
      - remove obvious timestamps/UUID tokens
      - strip repeated punctuation
      - trim to a reasonable word length (prefer 8–16 words)
      - use fallback if result is too short
    """
    if not raw_title:
        return fallback or "Generated Paper"

    t = str(raw_title).strip()

    # Remove fenced code block wrappers if present
    fence_match = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", t, flags=re.S)
    if fence_match:
        t = fence_match.group(1).strip()

    # Remove UUIDs and hex-like tokens and date-like patterns
    t = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "", t)
    t = re.sub(r"\b[0-9a-fA-F]{8,}\b", "", t)
    t = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", t)
    t = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s*\d{1,2}(?:,\s*\d{4})?\b", "", t, flags=re.I)

    # Remove repeated punctuation and extra whitespace
    t = re.sub(r'[_\-]{2,}', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = t.strip(" \n\t\"'.,:;")

    # If LLM returned multiple sentences, take the first as title
    if '.' in t:
        t = t.split('.')[0]

    # Limit word count (prefer between 8 and 16)
    words = t.split()
    if len(words) > 16:
        t = " ".join(words[:14])
    # If too short and fallback available, prefer fallback
    if len(words) < 3 and fallback:
        t = fallback

    t = t.strip()
    if not t:
        return fallback or "Generated Paper"
    return t


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

    Note: query_chunks is imported lazily to avoid circular import issues.
    """
    llm = _get_llm(model_name)

    # Allowed order and canonical section names
    section_order = [
        "title", "abstract", "introduction", "literature_review", "methods",
        "experiments", "results", "conclusion", "references"
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
            if key in {"literature-review", "literature_review", "literaturereview"}:
                key = "literature_review"
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

    # Build a short, high-signal facts summary for the LLM
    facts_summary = _build_facts_summary(facts)

    # RAG: retrieve relevant chunks and create a compact context block (lazy import)
    rag_context = ""
    if rag_context_override:
        rag_context = rag_context_override
    elif use_rag:
        queries = []
        if facts.get("task"):
            queries.append(facts["task"])
        # include summary as a high priority query if present
        if facts_summary:
            queries.insert(0, facts_summary)
        queries.extend(facts.get("datasets", [])[:2])
        queries.extend(facts.get("methods", [])[:2] if facts.get("methods") else [])

        if queries:
            try:
                # lazy import to avoid circular dependency at module import time
                from services.rag_retriever import query_chunks as _query_chunks
                chunks = _query_chunks(queries, top_k=top_k)
            except Exception as e:
                # Don't fail the whole generation if retriever is unavailable
                print(f"[RAG] query_chunks failed or not available: {e}")
                chunks = []

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
        f"Notebook facts (short): {facts_summary}\n\n"
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

        # Special handling for title: stronger prompt + sanitize
        if section == "title":
            prompt = (
                f"{common_context}\n"
                "Write only a clear, descriptive research paper title (8–16 words) summarizing the main contribution.\n"
                "Avoid dates, file names, UUIDs, parentheses with dataset names, and do not include citations.\n"
                "Return only the title text (no punctuation decorations or Markdown headers).\n"
            )
            raw_title = _safe_invoke(llm, prompt)
            fallback_title = ""
            if facts.get("markdown"):
                for md_line in facts.get("markdown", []):
                    if md_line and isinstance(md_line, str) and len(md_line.strip()) > 10:
                        fallback_title = md_line.strip().splitlines()[0][:120]
                        break
            sanitized = _sanitize_title(raw_title, fallback=fallback_title or "Generated Paper")
            produced["title"] = sanitized
            print(f"[LLM] Produced title: {sanitized!r}")
            continue

        instr = SECTION_INSTRUCTIONS.get(section, "")

        if section == "references":
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
                    m = re.match(r"^\[CITATION:\s*(.+?)\]$", token, flags=re.I)
                    if m:
                        token = m.group(1).strip()

                    doi_norm = _normalize_doi(token)
                    matched = None
                    if doi_norm and doi_norm in candidate_doi_map:
                        matched = candidate_doi_map[doi_norm]
                    else:
                        token_norm = _normalize_title_for_match(token)
                        for t_norm, p in candidate_title_map:
                            if token_norm in t_norm or t_norm in token_norm:
                                matched = p
                                break
                    if matched:
                        use = (matched.get("doi") or matched.get("title") or token).strip()
                        final_refs.append(use)
                    else:
                        # skip external refs not in candidate list
                        continue
            else:
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
            print(f"[LLM] Produced references placeholders: {produced['references'][:200]}")
        else:
            # textual section (including literature_review)
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

            # For literature_review we can add an explicit guidance about length / synthesis
            section_label = section.replace('_', ' ').title()
            prompt = (
                f"{common_context}\n"
                f"Write the {section_label} section.\n"
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

            # Remove citation placeholders from results and conclusion explicitly
            if section in ("results", "conclusion"):
                text = _remove_citation_placeholders(text).strip()
                text = re.sub(r"\[CITATION:[^\]]+\]", "", text, flags=re.I)
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\s+([.,;:])", r"\1", text)

            produced[section] = text
            preview = (text[:200] + '...') if len(text) > 200 else text
            print(f"[LLM] Produced {section}: {preview!r}")

    return produced
