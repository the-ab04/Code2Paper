# backend/services/llm_generator.py

from typing import Dict, Any, List, Optional
from config import GROQ_API_KEY, MODEL_NAME
# NOTE: langchain_groq and query_chunks are imported lazily inside functions to
# avoid circular imports and reduce cold-start time.
import json
import re
import os

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
        "Avoid citations, dataset names in parentheses, dates, file names, UUIDs, or punctuation-heavy qualifiers."
    ),
    "abstract": (
    "Write a single-paragraph abstract (120–220 words) that summarizes the problem, approach, key results, "
    "and main contributions. Do NOT include citation placeholders in the abstract; remove them if present. "
    "Do NOT mention or refer to any figures, figure numbers, figure placeholders, or captions — the abstract must "
    "be self-contained and free of figure-related details."
    ),
    "introduction": (
        "Write an Introduction that motivates the problem, provides brief background, identifies the research gap, "
        "and lists the contributions of this work. Provide sufficient context so a reader unfamiliar with the code can follow. "
        "Target length: ~300–600 words. Use citation placeholders [CITATION: query] when referencing prior work; prefer DOIs."
    ),
    "literature_review": (
        "Write a Literature Review that synthesizes prior work relevant to the task. Group related works by theme/approach "
        "(3–6 themes), compare representative methods/benchmarks for each theme, and highlight remaining gaps that justify the present work. "
        "Aim to be comprehensive and analytic rather than a short list. Target length: ~600–1000 words. Insert citation placeholders [CITATION: query] where appropriate (prefer DOIs)."
    ),
    "methods": (
        "Write a Methods section describing the model/architecture, datasets, preprocessing, training procedures, hyperparameters, "
        "and implementation details necessary to reproduce results. Be specific about shapes, training loops, optimizer, learning rates, and seeds where known. "
        "Target length: ~500–900 words. Use citation placeholders [CITATION: query] for standard components/datasets when relevant."
    ),
    "experiments": (
        "Write an Experiments section describing datasets and splits, evaluation protocol, baselines, metrics, and the experimental procedure "
        "(runs/seeds/hyperparameter sweeps). Provide enough detail to reproduce the experiments. Target length: ~400–800 words. Use citation placeholders only when comparing to prior work."
    ),
    "results": (
    "Write a concise Results section (150–300 words) summarizing key quantitative and qualitative findings. "
    "Focus only on main performance metrics, trends, and observations. "
    "Avoid repeating methodology or dataset descriptions. "
    "Refer to figures only briefly and only when necessary (e.g., 'Figure 1 shows model accuracy trends'), "
    "without describing every figure in detail. "
    "Do NOT include raw code outputs, redundant explanations, or any placeholder text."
    ),
    "conclusion": (
        "Write a short Conclusion (2–4 sentences) summarizing the main takeaways and proposing future directions. "
        "Do NOT include citation placeholders in the conclusion; remove them if present. Target length: ~40–120 words."
    ),
    "references": (
        "Return a JSON array of citation placeholders (strings). Each element should be a short query suitable for CrossRef lookup or the downstream citation resolver, e.g. a paper title or DOI. Prefer DOIs if available. Return only a JSON array, nothing else."
    ),
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
                return [str(x).strip() for x in parsed if x is not None]
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

    # Prefer structured output_metrics if present (facts["output_metrics"] is a dict)
    output_metrics = facts.get("output_metrics") or facts.get("final_metrics") or {}
    if isinstance(output_metrics, dict) and output_metrics:
        parts.append("Output metrics: " + ", ".join(f"{k}={v}" for k, v in list(output_metrics.items())[:6]))
    else:
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


# helper to build figures block for prompts
def _format_figures_block(figures: List[Dict[str, Any]]) -> str:
    """
    Build a short human-readable figures block:
      [1] path: <basename> | hint: <caption_hint> | cell: <cell_index>
    """
    if not figures:
        return ""
    lines = ["=== Notebook Figures (selected) ==="]
    for idx, f in enumerate(figures, start=1):
        p = f.get("path") or ""
        hint = (f.get("caption_hint") or "")[:200]
        ci = f.get("cell_index")
        lines.append(f"[{idx}] Path: {os.path.basename(p)} | Hint: {hint or '(none)'} | Cell: {ci}")
    lines.append(
        "For each figure above, you may reference it in Results/Methods if relevant. "
        "When asked, produce one concise, one-sentence caption per figure suitable for inclusion under the image."
    )
    return "\n".join(lines) + "\n"


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
    md = "\n\n".join(facts.get("markdown", []))[:]
    code_hint = "\n".join(facts.get("code", [])[:6])
    logs = "\n".join(facts.get("logs", [])[:10])
    datasets = ", ".join(facts.get("datasets", []) or [])
    metrics = ", ".join(facts.get("metrics", []) or [])

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
        queries.extend(facts.get("datasets", [])[:2] if facts.get("datasets") else [])
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

    # Figures block (if available)
    figures_block = _format_figures_block(facts.get("figures") or [])

    # include output_metrics in context (stringified)
    output_metrics = facts.get("output_metrics") or facts.get("final_metrics") or {}
    output_metrics_str = ""
    if isinstance(output_metrics, dict) and output_metrics:
        output_metrics_str = " | ".join(f"{k}={v}" for k, v in output_metrics.items())

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
        f"Output metrics: {output_metrics_str}\n\n"
        f"{figures_block}\n"
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
            if figures_block:
                extra = (extra + "\n" if extra else "") + (
                    "You may reference the notebook figures listed above when describing results or methods. "
                    "When describing a figure, use its index (e.g., 'Figure [1]') and be concise."
                )
            if output_metrics:
                extra = (extra + "\n" if extra else "") + (
                    "Use the reported output metrics above when summarizing results. Reference numeric values exactly when appropriate."
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

    # --- New: Request one-sentence captions for notebook figures if any ---
    figs = facts.get("figures") or []
    if figs:
        try:
            # Build a richer figures prompt by including: caption_hint, code snippet, output first line, and run metrics
            figs_list_block = []
            # helper to get output snippet for a cell
            outputs_by_cell = {}
            for o in facts.get("outputs", []) or []:
                ci = o.get("cell_index")
                if ci is None:
                    continue
                # pick first non-empty line as representative
                text = (o.get("text") or "") + " " + (o.get("error") or "")
                first = ""
                for ln in (text or "").splitlines():
                    ln2 = ln.strip()
                    if ln2:
                        first = ln2
                        break
                outputs_by_cell[ci] = first

            code_blocks = facts.get("code") or []

            # create per-figure descriptive entries
            for idx, f in enumerate(figs, start=1):
                fname = os.path.basename(f.get("path") or "")
                hint = f.get("caption_hint") or ""
                ci = f.get("cell_index")
                code_snip = ""
                if isinstance(code_blocks, list) and ci is not None and 0 <= ci < len(code_blocks):
                    cb = code_blocks[ci] or ""
                    # include only short snippet (first 2-4 lines)
                    lines = [ln for ln in cb.splitlines() if ln.strip()]
                    code_snip = " | code: " + " ".join(lines[:4]) if lines else ""
                    # trim to reasonable length
                    if len(code_snip) > 300:
                        code_snip = code_snip[:300] + "..."
                output_first = outputs_by_cell.get(ci, "")
                metrics = facts.get("output_metrics") or facts.get("final_metrics") or {}
                metrics_str = ""
                if isinstance(metrics, dict) and metrics:
                    metrics_str = " | metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items())

                entry = f"[{idx}] {fname} | hint: {hint or '(none)'} | cell: {ci}"
                if code_snip:
                    entry += f" {code_snip}"
                if output_first:
                    entry += f" | output: {output_first[:200]}"
                if metrics_str:
                    entry += f" {metrics_str}"
                figs_list_block.append(entry)

            figs_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Below are the notebook figures selected as potentially useful for the paper, with context (filename, hint, code snippet, output snippet, and run metrics):\n\n"
                f"{chr(10).join(figs_list_block)}\n\n"
                "For each figure above produce a single concise (one-sentence) caption suitable for a research paper. "
                "Captions should be factual, describe the key content of the figure, may mention numeric metrics if the figure visualizes them, "
                "and be written in plain English. Return ONLY a JSON array of strings (one string per figure in the same order). "
                "Example: [\"Caption 1\", \"Caption 2\"]\n"
            )

            raw_caps = _safe_invoke(llm, figs_prompt)
            caps_list = _extract_json_array(raw_caps)

            # If the model returned fewer captions than figures, attempt a robust fallback:
            if len(caps_list) < len(figs):
                # Try to extract newline-separated plain lines as backup (very tolerant)
                extra_lines = [ln.strip() for ln in str(raw_caps).splitlines() if ln.strip()]
                for ln in extra_lines:
                    if ln not in caps_list and len(caps_list) < len(figs):
                        # remove surrounding quotes/brackets
                        ln_clean = ln.strip().strip('`"\' ,[]')
                        if ln_clean:
                            caps_list.append(ln_clean)
                # finally pad with hints
            # Normalize length: ensure we have same number as figs (truncate or pad with fallback)
            final_caps: List[str] = []
            for i, f in enumerate(figs):
                if i < len(caps_list):
                    c = caps_list[i].strip()
                    # ensure single-line short caption
                    c = re.sub(r"\s+", " ", c).strip()
                    # if caption is too long, shorten to first sentence
                    if len(c.split()) > 40:
                        # take first sentence if available
                        sents = re.split(r'(?<=[.!?])\s+', c)
                        c = sents[0] if sents and sents[0] else " ".join(c.split()[:30]) + "..."
                    final_caps.append(c)
                else:
                    # fallback: prefer caption_hint, otherwise output snippet, otherwise filename
                    hint = f.get("caption_hint") or ""
                    ci = f.get("cell_index")
                    output_snip = outputs_by_cell.get(ci, "")
                    fallback_caption = ""
                    if hint:
                        fallback_caption = f"{hint}."
                    elif output_snip:
                        # first non-empty output line truncated
                        fallback_caption = (output_snip[:140] + "...") if len(output_snip) > 140 else output_snip
                    else:
                        fallback_caption = os.path.basename(f.get("path") or "")
                    # ensure it's one sentence
                    fallback_caption = re.sub(r"\s+", " ", fallback_caption).strip()
                    if not fallback_caption.endswith("."):
                        fallback_caption = fallback_caption.rstrip(".") + "."
                    final_caps.append(fallback_caption)

            produced["figure_captions"] = json.dumps(final_caps, ensure_ascii=False)
            print(f"[LLM] Produced {len(final_caps)} figure captions.")
        except Exception as e:
            # non-fatal: build deterministic captions from hints/outputs
            print(f"[Figure Captions] failed to produce captions via LLM: {e}")
            deterministic_caps = []
            outputs_by_cell = {}
            for o in facts.get("outputs", []) or []:
                ci = o.get("cell_index")
                text = (o.get("text") or "") + " " + (o.get("error") or "")
                first = ""
                for ln in (text or "").splitlines():
                    ln2 = ln.strip()
                    if ln2:
                        first = ln2
                        break
                outputs_by_cell[ci] = first
            for f in figs:
                hint = f.get("caption_hint") or ""
                ci = f.get("cell_index")
                out_snip = outputs_by_cell.get(ci, "")
                if hint:
                    cap = hint
                elif out_snip:
                    cap = out_snip[:160]
                else:
                    cap = os.path.basename(f.get("path") or "")
                cap = re.sub(r"\s+", " ", cap).strip()
                if not cap.endswith("."):
                    cap = cap.rstrip(".") + "."
                deterministic_caps.append(cap)
            produced["figure_captions"] = json.dumps(deterministic_caps, ensure_ascii=False)

    return produced
