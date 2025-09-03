from typing import Dict, Any
from config import GROQ_API_KEY, MODEL_NAME
from langchain_groq import ChatGroq
import json, re

SYSTEM_PROMPT = (
    "You are an assistant that converts Jupyter notebooks into structured academic paper sections. "
    "Write concise, factual text in formal academic tone. "
    "Include methods, experiments, results, and limitations. "
    "For citations, insert placeholders in the format [CITATION: query] where query is the title or DOI. "
    "Return ONLY valid JSON (no markdown, no code fences)."
)

def _get_llm(model_name: str | None = None):
    """Initialize the Groq LLM client."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in config.")
    return ChatGroq(model=model_name or MODEL_NAME, temperature=0.2, groq_api_key=GROQ_API_KEY)

def _extract_json(text: str) -> str:
    """Extract a clean JSON string from LLM output."""
    # Remove markdown fences like ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)

    # Extract from first '{' to last '}'
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Normalize quotes
    text = (text.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'"))
    return text.strip()

def generate_sections(facts: Dict[str, Any], model_name: str | None = None) -> Dict[str, str]:
    """Generate structured academic paper sections with citation placeholders."""
    llm = _get_llm(model_name)

    # Prepare notebook content
    md = "\n\n".join(facts.get("markdown", []))[:6000]
    code_hint = "\n".join(facts.get("code", [])[:3])
    logs = "\n".join(facts.get("logs", [])[:10])

    # Prompt requiring JSON output
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Notebook markdown (truncated):\n{md}\n\n"
        f"Sample code (truncated):\n{code_hint}\n\n"
        f"Training/metrics logs (sample):\n{logs}\n\n"
        "Write the following sections with citations as placeholders [CITATION: query]:\n"
        "1. Title\n2. Abstract\n3. Introduction\n4. Methods\n5. Experiments\n"
        "6. Results\n7. Discussion\n8. Conclusion\n9. References (array of queries for citations)\n\n"
        "Output strictly as JSON with keys: title, abstract, introduction, methods, experiments, "
        "results, discussion, conclusion, references."
    )

    # Invoke LLM
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp))
    cleaned = _extract_json(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        data = json.loads(cleaned)

    # Normalize result
    result: Dict[str, str] = {}
    for key in ["title", "abstract", "introduction", "methods", "experiments", "results", "discussion", "conclusion"]:
        result[key] = str(data.get(key, "")).strip()

    refs = data.get("references", [])
    if isinstance(refs, list):
        result["references"] = "\n".join([str(r).strip() for r in refs if r])
    else:
        result["references"] = str(refs or "").strip()

    return result
