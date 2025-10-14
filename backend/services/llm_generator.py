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
        "Insert citation placeholders as [CITATION: query] when referencing prior work."
    ),
    "methods": (
        "Write a Methods section describing the model/architecture, datasets, training, hyperparameters, and any "
        "implementation details necessary to reproduce the work. Use citation placeholders [CITATION: query] where "
        "appropriate to reference methods from the literature."
    ),
    "experiments": (
        "Write an Experiments section describing the evaluation protocol, datasets/splits, baselines, metrics, and "
        "experimental setup. Use citation placeholders where comparing to prior works."
    ),
    "results": (
        "Write a Results section summarizing quantitative and qualitative findings, include key tables/metrics "
        "phrases (no raw tables), and mention notable observations. Use citation placeholders where referencing "
        "benchmarks or prior reported numbers."
    ),
    "discussion": (
        "Write a Discussion: interpret results, discuss strengths and weaknesses, possible causes of behavior, "
        "and practical implications. Use citation placeholders as needed."
    ),
    "conclusion": (
        "Write a short Conclusion (2–4 sentences) summarizing the work and listing future directions."
    ),
    # 'references' is handled separately and must return valid JSON array
    "references": (
        "Return a JSON array of citation placeholders (strings). Each item should be a short query suitable for "
        "CrossRef lookup or the format used by downstream citation resolver, e.g. a paper title or DOI. "
        "Return only JSON array, nothing else."
    )
}

# patterns
CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]")

# === LLM helper ===
def _get_llm(model_name: Optional[str] = None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in config.")
    return ChatGroq(model=model_name or MODEL_NAME, temperature=0.15, groq_api_key=GROQ_API_KEY)


def _safe_invoke(llm, prompt: str) -> str:
    """Call the LLM and return assistant content as str (robust)."""
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return str(raw)


def _extract_json_array(text: str) -> List[str]:
    """
    Extract a JSON array from a string and return Python list.
    Accepts arrays like ["a","b"] possibly surrounded by text or code fences.
    """
    # remove code fences first
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        # fallback: try to find lines that look like placeholders
        candidates = CITATION_PATTERN.findall(text)
        return [c.strip() for c in candidates] if candidates else []

    arr_text = text[start:end + 1]

    # try load json; if fails, attempt safe repairs
    try:
        parsed = json.loads(arr_text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
        return []
    except Exception:
        # remove trailing commas and non-JSON tokens
        arr_text = re.sub(r",\s*([}\]])", r"\1", arr_text)
        arr_text = arr_text.replace("\n", " ")
        # extract quoted items
        items = re.findall(r'"([^"]+)"', arr_text)
        if items:
            return [i.strip() for i in items]
        # last resort: split on commas inside brackets
        inner = arr_text.strip()[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
        return parts


def _remove_citation_placeholders(text: str) -> str:
    """Remove any [CITATION: ...] placeholders from text."""
    return CITATION_PATTERN.sub("", text)


# === Main public function ===
def generate_sections(
    facts: Dict[str, Any],
    model_name: Optional[str] = None,
    use_rag: bool = True,
    top_k: int = 5
) -> Dict[str, str]:
    """
    Generate the paper sections. Returns a dict with keys:
    title, abstract, introduction, methods, experiments, results, discussion, conclusion, references

    - Abstract will have citation placeholders removed.
    - references is a newline-separated string of placeholders (one per line).
    """
    llm = _get_llm(model_name)

    # Prepare notebook context snippet
    md = "\n\n".join(facts.get("markdown", []))[:4000]
    code_hint = "\n".join(facts.get("code", [])[:3])
    logs = "\n".join(facts.get("logs", [])[:10])
    datasets = ", ".join(facts.get("datasets", []))
    metrics = ", ".join(facts.get("metrics", []))

    # RAG: retrieve relevant chunks and create a compact context block
    rag_context = ""
    if use_rag:
        queries = []
        # build a small query list from facts
        if facts.get("task"):
            queries.append(facts["task"])
        queries.extend(facts.get("datasets", [])[:2])
        # optional methods found during parsing
        queries.extend(facts.get("methods", [])[:2] if facts.get("methods") else [])

        if queries:
            chunks = query_chunks(queries, top_k=top_k)
            if chunks:
                rag_context = "=== Retrieved Literature Chunks ===\n"
                for idx, ch in enumerate(chunks, start=1):
                    src = ch.get("doi") or ch.get("title") or f"Source {idx}"
                    snippet = ch.get("text") or ch.get("chunk") or ch.get("chunk", "")
                    # keep snippet short
                    snippet_short = (snippet[:800] + "...") if len(snippet) > 800 else snippet
                    rag_context += f"[{idx}] {snippet_short} (Source: {src})\n"

    # Construct a common context header inserted into each prompt
    common_context = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Notebook facts (sample):\n{md}\n\n"
        f"Code sample:\n{code_hint}\n\n"
        f"Training logs:\n{logs}\n\n"
        f"Datasets: {datasets}\n"
        f"Metrics: {metrics}\n\n"
        f"{rag_context}\n"
    )

    # Sections order we will produce
    section_order = [
        "title", "abstract", "introduction", "methods",
        "experiments", "results", "discussion", "conclusion", "references"
    ]

    produced: Dict[str, str] = {}

    # iterate and produce each section independently
    for section in section_order:
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
            # store as newline separated placeholders (downstream expects placeholders like: [CITATION: query])
            # ensure we wrap items to placeholders so citation_manager finds them if needed
            placeholder_lines = []
            for r in refs_list:
                item = r.strip()
                if not item:
                    continue
                # if user/LLM already returned something like "[CITATION: ...]" keep it; otherwise wrap
                if re.match(r"^\[CITATION:", item, re.I):
                    placeholder_lines.append(item)
                else:
                    placeholder_lines.append(f"[CITATION: {item}]")
            produced["references"] = "\n".join(placeholder_lines)
        else:
            # normal textual sections
            prompt = (
                f"{common_context}\n"
                f"Write the {section.upper()} section.\n"
                f"{instr}\n\n"
                "Return only the section text (no surrounding JSON or markdown fences)."
            )
            raw = _safe_invoke(llm, prompt)
            text = str(raw).strip()

            # Clean: remove surrounding code fences if present
            fence_match = re.search(r"```(?:\w+)?\n(.*)\n```$", text, flags=re.S)
            if fence_match:
                text = fence_match.group(1).strip()

            # For abstract only: remove citation placeholders entirely (user requested)
            if section == "abstract":
                text = _remove_citation_placeholders(text).strip()
                # Also fix double spaces and stray punctuation left by removal
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\s+([.,;:])", r"\1", text)

            produced[section] = text

    # Ensure all required keys exist
    for key in section_order:
        if key not in produced:
            produced[key] = ""

    # Format references as newline string (already done)
    return produced
