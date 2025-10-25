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

# Tunables
MAX_CANDIDATE_PAPERS = int(os.getenv("MAX_CANDIDATE_PAPERS", "10"))  # how many candidate papers we pass to LLM

CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


class GenerateRequest(BaseModel):
    """Request body schema for /generate/{run_id}"""
    sections: Optional[List[str]] = None
    use_rag: bool = True


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
    except Exception as e:
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
      7. render DOCX, update run status and return download URL
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

    # Step 2: Build better queries using facts summary + datasets, methods, metrics, model hints
    queries: List[str] = []

    # code_parser may produce a short summary or 'facts_summary' - prefer that
    if facts.get("summary"):
        queries.append(facts["summary"])
    # datasets, methods, frameworks, model_defs, metrics
    queries.extend([d for d in (facts.get("datasets") or []) if d])
    queries.extend([m for m in (facts.get("methods") or []) if m])
    queries.extend([m for m in (facts.get("model_defs") or []) if m])
    queries.extend([f for f in (facts.get("frameworks") or []) if f])
    queries.extend([m for m in (facts.get("metrics") or []) if m])

    # fallback
    if not queries:
        queries = ["machine learning"]

    external_papers: List[Dict[str, Any]] = []
    if payload.use_rag:
        try:
            # Persist & index candidate papers (returns persisted entries when possible)
            persisted = find_and_persist_papers(queries, run.id, db, max_results=10)
            external_papers = persisted or []
            logger.info("Persisted and indexed %d external papers for run %d", len(external_papers), run.id)
        except Exception as e:
            logger.exception("[Paper Finder] find_and_persist_papers failed, falling back to find_papers()")
            try:
                external_papers = find_papers(queries, max_results=5)
            except Exception as e2:
                logger.exception("[Paper Finder] find_papers() failed as well")
                external_papers = []

    # Step 3: Generate sections with LLM
    # Late-import generate_sections to avoid circular import with code_parser -> llm_generator
    try:
        llm_mod = importlib.import_module("services.llm_generator")
        generate_sections_fn = getattr(llm_mod, "generate_sections")
    except Exception as e:
        logger.exception("failed to import services.llm_generator.generate_sections")
        raise HTTPException(status_code=500, detail="LLM generation helper not available (import error)")

    # Update run status to 'generating'
    try:
        crud.update_run_status(db, run.id, status="generating")
    except Exception:
        logger.warning("Failed to set run status to 'generating' for run %s", run.id)

    try:
        # Limit candidate papers passed to the LLM to a reasonable number
        candidate_list = external_papers[:MAX_CANDIDATE_PAPERS] if external_papers else None

        if candidate_list:
            logger.info("Calling LLM with %d candidate papers (limited to %d)", len(candidate_list), MAX_CANDIDATE_PAPERS)
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
    # find_and_persist_papers likely already persisted items (returned with 'db_id').
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
                except Exception as e:
                    logger.warning("Failed to add candidate paper to DB (non-fatal): %s", e)
    except Exception as e:
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
        except Exception as e:
            logger.exception("[Citation Manager] error while enriching references: %s", e)

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

    # Step 6: Render DOCX to outputs
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
        render_docx(sections, out_file=out_path, title=title_for_doc)
    except Exception:
        logger.exception("[File Generator] error while rendering DOCX")
        crud.update_run_status(db, run.id, status="failed")
        raise HTTPException(status_code=500, detail="Failed to render document")

    # Step 7: Update DB run status
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
