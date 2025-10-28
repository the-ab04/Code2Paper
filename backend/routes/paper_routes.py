# backend/routes/paper_routes.py

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import shutil
import os
import uuid
import re
import importlib
import logging
import json

from db.base import get_db
from db import crud, schemas
from services.code_parser import parse_notebook
# NOTE: we intentionally do NOT import generate_sections at module import time to avoid circular imports.
from services.file_generator import render_docx
from services.citation_manager import enrich_references
from services.paper_finder import find_papers, find_and_persist_papers
from services.reset_manager import full_reset
import config

logger = logging.getLogger(__name__)

# Storage directories (config-driven)
STORAGE_ROOT = getattr(config, "STORAGE_DIR", "storage")
UPLOAD_DIR = os.path.join(STORAGE_ROOT, "uploads")
OUTPUT_DIR = os.path.join(STORAGE_ROOT, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tunables (defaults can be overridden via env or request body)
DEFAULT_MAX_CANDIDATE_PAPERS = int(os.getenv("MAX_CANDIDATE_PAPERS", "10"))  # how many candidate papers we pass to LLM
DEFAULT_MAX_FIGURES = int(os.getenv("MAX_FIGURES", "6"))  # how many figures to include in DOCX
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", str(8 * 1024)))  # 8KB default minimum to consider image "real"

CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


class GenerateRequest(BaseModel):
    """Request body schema for /generate/{run_id}"""
    sections: Optional[List[str]] = None
    use_rag: bool = True
    max_papers: Optional[int] = None      # override how many candidate papers to pass to LLM
    max_figures: Optional[int] = None     # override how many figures to include in final document


router = APIRouter(prefix="/api/paper", tags=["paper"])


@router.post("/upload", response_model=schemas.RunOut)
def upload_notebook(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload notebook/script and create a new run.
    (Current behavior: full reset before new run.)
    """
    # Reset system for a fresh run (keep current behavior, remove if undesired)
    try:
        full_reset(db)
    except Exception as e:
        logger.warning("full_reset failed or not desired: %s", e)

    if not (file.filename.endswith(".ipynb") or file.filename.endswith(".py")):
        raise HTTPException(status_code=400, detail="Only .ipynb or .py files are allowed")

    file_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    run = crud.create_run(db=db, input_file=save_path)
    return run


def _normalize_requested_sections(sections: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize client provided sections to canonical lowercase names or return None if not provided.
    Accepts common synonyms (e.g. 'methodology' -> 'methods', 'literature-review' -> 'literature_review').
    """
    if not sections:
        return None

    # canonical names supported by backend/LLM
    canonical = {
        "title",
        "abstract",
        "introduction",
        "methods",
        "experiments",
        "results",
        "conclusion",
        "references",
        "literature_review",
    }

    # synonyms mapping (lowercase keys)
    synonyms = {
        "methodology": "methods",
        "method": "methods",
        "literature-review": "literature_review",
        "literature review": "literature_review",
        "literaturereview": "literature_review",
        "lit_review": "literature_review",
        "litreview": "literature_review",
        "literature_review": "literature_review",
        "refs": "references",
        "ref": "references",
        "intro": "introduction",
        "results": "results",
        "experiment": "experiments",
    }

    out: List[str] = []
    for s in sections:
        if not s:
            continue
        s_raw = str(s).strip()
        # normalize separators: spaces and hyphens -> underscore, lowercased
        s_norm = s_raw.lower().replace("-", " ").replace("_", " ").strip()
        # compress multiple spaces
        s_norm = re.sub(r"\s+", " ", s_norm)

        # prefer synonyms first
        if s_norm in synonyms:
            cand = synonyms[s_norm]
            if cand in canonical:
                out.append(cand)
            continue

        # canonical check: try direct collapse to underscore form
        s_key = s_norm.replace(" ", "_")
        if s_key in canonical:
            out.append(s_key)
            continue

        # try again raw exact (fallback)
        if s_raw.lower() in canonical:
            out.append(s_raw.lower())
            continue

    # return None if empty so LLM uses default (full set)
    return out or None


def _select_useful_images(images: List[Dict[str, Any]], max_figures: int) -> List[Dict[str, Any]]:
    """
    Select up to `max_figures` useful images from the extracted images list using a heuristic:
      - prefer images that already have a caption
      - next prefer images with an explicit 'score' (higher better)
      - next prefer larger file size (above MIN_IMAGE_BYTES)
    Each image is expected to be a dict with at least 'path' and optional 'caption','score','cell_index','mime'.
    """
    if not images:
        return []

    def _size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

    norm = []
    seen_paths = set()
    for im in images:
        p = im.get("path")
        if not p:
            continue
        p_abs = os.path.abspath(p)
        if p_abs in seen_paths:
            continue
        if not os.path.exists(p_abs):
            continue
        seen_paths.add(p_abs)
        size = _size(p_abs)
        caption_flag = 1 if im.get("caption") else 0
        try:
            score_val = float(im.get("score")) if im.get("score") is not None else 0.0
        except Exception:
            score_val = 0.0
        size_flag = 1 if size >= MIN_IMAGE_BYTES else 0
        # keep original dict but normalize path to absolute
        item = dict(im)
        item["path"] = p_abs
        item["size"] = size
        norm.append((caption_flag, score_val, size_flag, size, item))

    # Sort descending by (caption_flag, score_val, size_flag, size)
    norm.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)

    selected = [t[-1] for t in norm[:max_figures]]
    return selected


@router.post("/generate/{run_id}")
def generate_paper(
    run_id: int,
    payload: GenerateRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    Generate paper sections for a run.
    Workflow:
      1. parse notebook -> facts
      2. discover candidate papers (persist & index) if use_rag True
      3. call LLM to generate requested sections (passing candidate_papers when available)
         - If candidate_papers exist, RAG is disabled so the LLM uses ONLY the candidate list.
      4. persist any remaining candidate papers (best-effort)
      5. enrich/resolve citation placeholders and persist citations
      6. ALWAYS assemble final References from DB citations (ordered)
      7. normalize/remove figure placeholders (final cleanup)
      8. render DOCX, update run status and return download URL
    """
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not run.input_file or not os.path.exists(run.input_file):
        raise HTTPException(status_code=400, detail="Input file is missing")

    # Step 1: Parse notebook → structured facts
    facts = parse_notebook(run.input_file)

    # Normalize requested sections
    requested_sections = _normalize_requested_sections(payload.sections)

    # Ensure 'title' is always in generation request
    if requested_sections is None:
        gen_requested = None
    else:
        if "title" not in requested_sections:
            gen_requested = ["title"] + requested_sections
        else:
            gen_requested = requested_sections

    # Step 2: Build better queries using facts summary + datasets, methods, metrics, model hints, final metrics
    queries: List[str] = []

    # code_parser may produce a short summary or 'facts_summary' - prefer that
    if facts.get("summary"):
        queries.append(facts["summary"])

    # include datasets, methods, frameworks, model_defs, metrics
    queries.extend([d for d in (facts.get("datasets") or []) if d])
    queries.extend([m for m in (facts.get("methods") or []) if m])
    queries.extend([m for m in (facts.get("model_defs") or []) if m])
    queries.extend([f for f in (facts.get("frameworks") or []) if f])
    queries.extend([m for m in (facts.get("metrics") or []) if m])

    # include final numeric metrics (if any) as short "metric=value" tokens to bias search
    final_metrics = facts.get("final_metrics") or {}
    if isinstance(final_metrics, dict):
        for k, v in final_metrics.items():
            try:
                queries.append(f"{k}={v}")
            except Exception:
                continue

    # fallback
    if not queries:
        queries = ["machine learning"]

    # Determine max candidate papers to request/persist
    max_papers = payload.max_papers if (payload.max_papers and payload.max_papers > 0) else DEFAULT_MAX_CANDIDATE_PAPERS

    external_papers: List[Dict[str, Any]] = []
    if payload.use_rag:
        try:
            # Persist & index candidate papers (returns persisted entries when possible)
            persisted = find_and_persist_papers(queries, run.id, db, max_results=max_papers)
            # find_and_persist_papers may return None on failure; guard it
            external_papers = persisted or []
            logger.info("Persisted and indexed %d external papers for run %d", len(external_papers), run.id)
        except Exception:
            logger.exception("[Paper Finder] find_and_persist_papers failed, falling back to find_papers()")
            try:
                external_papers = find_papers(queries, max_results=max(1, max_papers // 2))
            except Exception:
                logger.exception("[Paper Finder] find_papers() failed as well")
                external_papers = []

    # Step 3: Generate sections with LLM
    # Late-import generate_sections to avoid circular import with code_parser -> llm_generator
    try:
        llm_mod = importlib.import_module("services.llm_generator")
        generate_sections_fn = getattr(llm_mod, "generate_sections")
    except Exception:
        logger.exception("failed to import services.llm_generator.generate_sections")
        raise HTTPException(status_code=500, detail="LLM generation helper not available (import error)")

    # Update run status to 'generating'
    try:
        crud.update_run_status(db, run.id, status="generating")
    except Exception:
        logger.warning("Failed to set run status to 'generating' for run %s", run.id)

    try:
        # Limit candidate papers passed to the LLM
        candidate_list = external_papers[:max_papers] if external_papers else None

        if candidate_list:
            logger.info("Calling LLM with %d candidate papers", len(candidate_list))
            sections: Dict[str, str] = generate_sections_fn(
                facts,
                sections_to_generate=gen_requested,
                use_rag=False,  # rely only on candidate_papers
                candidate_papers=candidate_list,
                top_k=5,
            )
        else:
            logger.info("Calling LLM with use_rag=%s", payload.use_rag)
            sections = generate_sections_fn(
                facts,
                sections_to_generate=gen_requested,
                use_rag=payload.use_rag,
                top_k=5,
            )
    except TypeError:
        # Backward compatibility: function signature without candidate_papers
        sections = generate_sections_fn(
            facts,
            sections_to_generate=gen_requested,
            use_rag=payload.use_rag,
        )
    except Exception as e:
        logger.exception("[LLM Error] Generation failed for run %s: %s", run.id, e)
        crud.update_run_status(db, run.id, status="failed")
        raise HTTPException(status_code=500, detail=f"Section generation failed: {e}")

    # Debugging: log what the LLM produced
    try:
        produced_title = sections.get("title") if isinstance(sections.get("title"), str) else None
        logger.info("[LLM Output] Sections produced keys: %s", list(sections.keys()))
        logger.info("[LLM Output] Generated title: %s", produced_title)
        refs_preview = sections.get("references")
        if refs_preview:
            logger.debug("[LLM Output] References placeholder (raw):\n%s", (refs_preview[:1000]))
    except Exception:
        logger.exception("failed to log LLM outputs")

    # Step 4: Persist candidate papers that were returned by find_papers() but not already persisted.
    try:
        if external_papers:
            for p in external_papers:
                # if we have a db_id or id, assume persisted; skip
                if p.get("db_id") or p.get("id"):
                    continue
                try:
                    paper_schema = schemas.PaperBase(
                        title=p.get("title", "") or "No title",
                        authors=p.get("authors"),
                        year=int(p.get("year")) if p.get("year") else None,
                        venue=p.get("venue") or p.get("container") or p.get("source"),
                        doi=p.get("doi"),
                        url=p.get("url"),
                        pdf_path=p.get("pdf_path", None),
                    )
                    crud.add_paper(db, run.id, paper_schema)
                except Exception:
                    logger.warning("Failed to add candidate paper to DB (non-fatal)")
    except Exception:
        logger.exception("unexpected error while persisting candidate papers")

    # Step 5: Enrich references if there are placeholders
    placeholders_exist = False
    if isinstance(sections, dict):
        try:
            placeholders_exist = any(CITATION_PATTERN.search(str(text or "")) for text in sections.values())
        except Exception:
            placeholders_exist = False

    if placeholders_exist:
        try:
            sections = enrich_references(sections, run_id=run.id, db=db)
            unresolved = sections.get("_unresolved_placeholders")
            if unresolved:
                logger.info("[Citation Manager] Unresolved placeholders: %s", unresolved)
        except Exception:
            logger.exception("[Citation Manager] error while enriching references")

    # -------------------------
    # FIGURE PLACEHOLDERS HANDLING:
    # - Remove figure placeholders from abstract & conclusion entirely.
    # - Normalize figure placeholders in results & introduction to "Figure N".
    # - Keep citation placeholders "[CITATION: ...]" intact.
    # -------------------------

    def _normalize_figure_placeholders(text: Optional[str]) -> Optional[str]:
        """
        Normalize many common figure placeholder variants into 'Figure N' (capital F).
        Handles variants like:
           - Figure[1], Figure [1], Figure (1), Figure.1
           - Fig 1, Fig. 1, Fig[1], Fig:[1], Fig:(1)
           - fig1, fig.1, figure1
        Returns original text unchanged if None/empty.
        Conservative: only targets patterns that include 'fig'/'figure' token.
        """
        if not text or not isinstance(text, str):
            return text

        t = text

        # Pattern 1: forms with optional punctuation and optional brackets/parentheses around the number
        pat = re.compile(
            r'\b(?:fig(?:\.|ure)?|figure)\s*[:\.\-]?\s*[\(\[\s]*\s*(\d{1,4})\s*[\)\]\s]*',
            flags=re.I,
        )

        def _repl(m):
            num = m.group(1)
            return f"Figure {num}"

        t = pat.sub(_repl, t)

        # Pattern 2: tightly joined forms like "fig1" or "figure1"
        pat2 = re.compile(r'\b(?:fig(?:ure)?)(\d{1,4})\b', flags=re.I)
        t = pat2.sub(lambda m: f"Figure {m.group(1)}", t)

        # Collapse multiple spaces introduced by replacements
        t = re.sub(r'\s{2,}', ' ', t).strip()
        return t

    def _remove_figure_placeholders(text: Optional[str]) -> Optional[str]:
        """
        Remove common figure placeholder mentions entirely from text.
        Includes variants with 'fig'/'figure' and bracket/parenthesis/punctuation forms.
        Leaves other numeric bracket forms (like [1]) untouched unless prefixed by fig/figure token.
        """
        if not text or not isinstance(text, str):
            return text

        t = text

        # Remove token forms like "Figure [1]", "Fig. (1):", "fig1", "Figure.1"
        pat_remove = re.compile(
            r'\b(?:fig(?:\.|ure)?|figure)\s*[:\.\-]?\s*[\(\[\s]*\s*\d{1,4}\s*[\)\]\s]*\s*[:\.\-]?',
            flags=re.I,
        )
        t = pat_remove.sub("", t)

        # Also remove tight "fig1" / "figure1"
        pat_tight = re.compile(r'\b(?:fig(?:ure)?)(\d{1,4})\b', flags=re.I)
        t = pat_tight.sub("", t)

        # Collapse leftover punctuation and extra spaces
        t = re.sub(r'\s{2,}', ' ', t)
        # remove stray sequences like " ,", " .", " :" created by removal
        t = re.sub(r'\s+([,.:;])', r'\1', t)
        t = t.strip()
        return t

    try:
        if isinstance(sections, dict):
            # Remove from abstract & conclusion entirely
            if sections.get("abstract") and isinstance(sections.get("abstract"), str):
                sections["abstract"] = _remove_figure_placeholders(sections["abstract"])
            if sections.get("conclusion") and isinstance(sections.get("conclusion"), str):
                sections["conclusion"] = _remove_figure_placeholders(sections["conclusion"])

            # Normalize placeholders to "Figure N" in results and introduction (so they remain but canonical)
            for key in ("results", "introduction"):
                if sections.get(key) and isinstance(sections.get(key), str):
                    sections[key] = _normalize_figure_placeholders(sections[key])
    except Exception:
        logger.exception("failed to process figure placeholders in sections")

    # Build final References section from DB citations (deterministically)
    try:
        db_citations = crud.get_citations_by_run(db, run.id) or []

        def _cit_sort_key(c):
            idx = getattr(c, "index", None) or 0
            return (0 if idx and idx > 0 else 1, idx if idx else getattr(c, "id", 0))

        citations_sorted = sorted(db_citations, key=_cit_sort_key)

        refs_from_db: List[str] = []
        assigned_seq = 1
        for c in citations_sorted:
            paper = db.query(crud.models.Paper).filter(crud.models.Paper.id == c.paper_id).first()
            if not paper:
                continue
            if not paper.title:
                continue

            display_index = c.index if (c.index and c.index > 0) else assigned_seq
            if not (c.index and c.index > 0):
                assigned_seq += 1

            authors = paper.authors or "Unknown Author"
            title = paper.title or "Unknown Title"
            venue = paper.venue or ""
            year = getattr(paper, "year", "") or ""
            doi = paper.doi or ""
            ref_line = f"[{display_index}] {authors}, \"{title},\" {venue}, {year}. DOI: {doi}"
            refs_from_db.append(ref_line)

        if refs_from_db:
            sections["references"] = "\n".join(refs_from_db)
            logger.info("[References Assembly] Built %d references from DB for run %s", len(refs_from_db), run.id)
        else:
            if not sections.get("references"):
                sections["references"] = "No references available."
                logger.info("[References Assembly] No references found for run %s", run.id)
    except Exception:
        logger.exception("[References Assembly] failed to build references from DB")

    # If the user requested a subset of sections and explicitly omitted 'references', drop it
    if requested_sections is not None and "references" not in requested_sections:
        sections.pop("references", None)

    # Step 6: Select useful images and attach figure captions (if LLM produced them)
    try:
        # Prefer facts["figures"] (selected by parser) else facts["images"]
        figures_from_parser = []
        if isinstance(facts, dict):
            if facts.get("figures"):
                figures_from_parser = facts.get("figures") or []
            elif facts.get("images"):
                # fall back to raw saved images
                figures_from_parser = facts.get("images") or []

        max_figs = payload.max_figures if (payload.max_figures and payload.max_figures > 0) else DEFAULT_MAX_FIGURES
        selected_images = _select_useful_images(figures_from_parser, max_figs)

        # If LLM returned figure captions (e.g., sections["figure_captions"]), attach them by index
        fig_captions = sections.get("figure_captions") or sections.get("figureCaptions") or None
        if fig_captions:
            parsed_caps = []
            if isinstance(fig_captions, (list, tuple)):
                parsed_caps = [str(x).strip() for x in fig_captions]
            elif isinstance(fig_captions, str):
                try:
                    parsed = json.loads(fig_captions)
                    if isinstance(parsed, list):
                        parsed_caps = [str(x).strip() for x in parsed]
                except Exception:
                    # last-resort: extract quoted strings
                    parsed_caps = re.findall(r'"([^"]+)"', fig_captions) or re.findall(r"'([^']+)'", fig_captions)
            # attach parsed captions in order to selected images where missing
            for idx, im in enumerate(selected_images):
                if not im.get("caption") and idx < len(parsed_caps):
                    cap = parsed_caps[idx]
                    if cap:
                        im["caption"] = cap
    except Exception:
        logger.exception("failed to select/attach images")
        selected_images = []

    # Step 7: Render DOCX to outputs; pass selected images only
    file_id = str(uuid.uuid4())
    out_path = os.path.join(OUTPUT_DIR, f"{file_id}.docx")
    try:
        # Use the generated title if present. If not, use the uploaded filename (strip uuid prefix)
        if sections.get("title"):
            title_for_doc = sections.get("title")
        else:
            raw_name = os.path.splitext(os.path.basename(run.input_file))[0]
            if "_" in raw_name:
                title_for_doc = raw_name.split("_", 1)[1]
            else:
                title_for_doc = raw_name

        logger.info("[File] Rendering DOCX with title: %s -> %s", title_for_doc, out_path)

        # Pass selected images into render_docx so figures can be embedded
        render_docx(sections, out_file=out_path, title=title_for_doc, images=selected_images)
    except Exception:
        logger.exception("[File Generator] error while rendering DOCX")
        crud.update_run_status(db, run.id, status="failed")
        raise HTTPException(status_code=500, detail="Failed to render document")

    # Step 8: Update DB run status
    try:
        crud.update_run_status(db, run.id, status="completed", output_file=out_path)
    except Exception:
        logger.exception("Failed to update run status to completed")

    produced_sections = list(sections.keys())
    return {
        "run_id": run.id,
        "produced_sections": produced_sections,
        "download_url": f"/api/paper/download/{run.id}",
    }


@router.get("/download/{run_id}")
def download_paper(run_id: int, db: Session = Depends(get_db)):
    run = crud.get_run(db, run_id)
    if not run or not run.output_file or not os.path.exists(run.output_file):
        raise HTTPException(status_code=404, detail="Generated file not found")

    filename = os.path.basename(run.output_file)
    return FileResponse(
        path=run.output_file,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@router.get("/runs/{run_id}", response_model=schemas.RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = crud.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs", response_model=List[schemas.RunOut])
def list_runs(db: Session = Depends(get_db)):
    return db.query(crud.models.Run).all()
