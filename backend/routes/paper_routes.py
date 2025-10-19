# backend/routes/paper_routes.py

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import shutil, os, uuid, re

from db.base import get_db
from db import crud, schemas
from services.code_parser import parse_notebook
from services.llm_generator import generate_sections
from services.file_generator import render_docx
from services.citation_manager import enrich_references
from services.paper_finder import find_papers, find_and_persist_papers
from services.reset_manager import full_reset

router = APIRouter(prefix="/api/paper", tags=["paper"])

UPLOAD_DIR = "storage/uploads"
OUTPUT_DIR = "storage/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CITATION_PATTERN = re.compile(r"\[CITATION:\s*([^\]]+)\]", re.I)


class GenerateRequest(BaseModel):
    """Request body schema for /generate/{run_id}"""
    sections: Optional[List[str]] = None
    use_rag: bool = True


@router.post("/upload", response_model=schemas.RunOut)
def upload_notebook(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload notebook/script and create a new run.
    (Current behavior: full reset before new run.)
    """
    # Reset system for a fresh run (keep current behavior, remove if undesired)
    full_reset(db)

    if not (file.filename.endswith(".ipynb") or file.filename.endswith(".py")):
        raise HTTPException(status_code=400, detail="Only .ipynb or .py files are allowed")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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

    # synonyms mapping (lowecase keys)
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

        # canonical check: try direct collapse to underscore form (title-case in mapping uses underscores)
        s_key = s_norm.replace(" ", "_")
        if s_key in canonical:
            out.append(s_key)
            continue

        # try again raw exact (fallback)
        if s_raw.lower() in canonical:
            out.append(s_raw.lower())
            continue

    # return None if empty so LLM uses default full set
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

    # === IMPORTANT: Ensure 'title' is always generated by the LLM ===
    # We will ask the LLM for 'title' even if the user did not request it explicitly.
    # Internally we build `gen_requested` which is passed to the LLM; the final `sections`
    # will still be returned and we'll always include the generated title in the doc.
    if requested_sections is None:
        # None means "generate all", so pass None to generate_sections
        gen_requested = None
    else:
        # ensure title is the first element so LLM produces it
        if "title" not in requested_sections:
            gen_requested = ["title"] + requested_sections
        else:
            gen_requested = requested_sections

    # Step 2: Determine & persist candidate papers (before generation) if RAG requested
    queries = facts.get("datasets", []) + facts.get("metrics", [])
    if not queries:
        queries = ["machine learning"]

    external_papers: List[Dict[str, Any]] = []
    if payload.use_rag:
        try:
            persisted = find_and_persist_papers(queries, run.id, db, max_results=15)
            external_papers = persisted or []
            print(f"✅ Persisted and indexed {len(external_papers)} external papers for run {run.id}")
        except Exception as e:
            print(f"[Paper Finder] find_and_persist_papers failed: {e} — falling back to find_papers()")
            try:
                external_papers = find_papers(queries, max_results=5)
            except Exception as e2:
                print(f"[Paper Finder] find_papers() failed as well: {e2}")
                external_papers = []

    # Step 3: Generate sections with LLM
    # If we have persisted candidate papers, we force the LLM to use ONLY those candidates:
    #  - pass candidate_papers parameter
    #  - disable external RAG retrieval by setting use_rag=False to avoid mixing in other sources
    try:
        if external_papers:
            print(f"✅ Calling LLM with {len(external_papers)} candidate papers (RAG disabled).")
            sections: Dict[str, str] = generate_sections(
                facts,
                sections_to_generate=gen_requested,
                use_rag=False,  # disable external retrieval, rely only on candidate_papers
                candidate_papers=external_papers
            )
        else:
            # No candidate list: follow requested use_rag flag
            print("ℹ️ No candidate papers persisted. Calling LLM with use_rag=", payload.use_rag)
            sections = generate_sections(
                facts,
                sections_to_generate=gen_requested,
                use_rag=payload.use_rag
            )
    except TypeError:
        # backward compatibility: generate_sections may not accept candidate_papers
        sections = generate_sections(
            facts,
            sections_to_generate=gen_requested,
            use_rag=payload.use_rag
        )
    except Exception as e:
        # Surface LLM invocation errors as 500 so the front-end can handle gracefully
        print(f"[LLM Error] Generation failed for run {run.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Section generation failed: {e}")

    # Debugging: log what the LLM produced (especially the title)
    try:
        produced_title = sections.get("title") if isinstance(sections.get("title"), str) else None
        print(f"[LLM Output] Sections produced keys: {list(sections.keys())}")
        print(f"[LLM Output] Generated title: {produced_title!r}")
        # If references placeholders exist, print a preview
        refs_preview = sections.get("references")
        if refs_preview:
            print(f"[LLM Output] References placeholder (raw):\n{refs_preview[:1000]}")
    except Exception as e:
        print(f"[Debug] failed to log LLM outputs: {e}")

    # Step 4: Persist candidate papers if they weren't persisted earlier
    # (crud.add_paper is idempotent w.r.t DOI, so this is safe.)
    try:
        if external_papers:
            for p in external_papers:
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
                    print(f"[DB] Failed to add candidate paper: {e}")
    except Exception as e:
        print(f"[Paper persistence] unexpected error: {e}")

    # Step 5: Enrich references if there are placeholders
    # Ensure we convert values to strings when testing for placeholders to avoid type issues
    placeholders_exist = any(CITATION_PATTERN.search(str(text or "")) for text in sections.values())
    if placeholders_exist:
        try:
            sections = enrich_references(sections, run_id=run.id, db=db)
            # If the citation manager attached debug info, print it
            unresolved = sections.get("_unresolved_placeholders")
            if unresolved:
                print(f"[Citation Manager] Unresolved placeholders: {unresolved}")
        except Exception as e:
            print(f"[Citation Manager] error while enriching references: {e}")

    # --- NEW STEP: Build final References section from DB citations (deterministically)
    try:
        db_citations = crud.get_citations_by_run(db, run.id) or []
        # Sort by citation.index (if present and >0), else by id to keep deterministic order.
        def _cit_sort_key(c):
            idx = getattr(c, "index", None) or 0
            # Put indexed citations first (index > 0), then unindexed by id
            return (0 if idx and idx > 0 else 1, idx if idx else getattr(c, "id", 0))

        citations_sorted = sorted(db_citations, key=_cit_sort_key)

        refs_from_db: List[str] = []
        assigned_seq = 1  # fallback numbering for citations without explicit index
        for c in citations_sorted:
            # ensure we have the latest paper record
            paper = db.query(crud.models.Paper).filter(crud.models.Paper.id == c.paper_id).first()
            if not paper:
                continue
            if not paper.title:
                continue  # skip incomplete DB rows

            # determine index to display: prefer c.index if >0 else assigned_seq (and advance assigned_seq)
            display_index = c.index if (c.index and c.index > 0) else assigned_seq
            if not (c.index and c.index > 0):
                assigned_seq += 1

            authors = paper.authors or "Unknown Author"
            title = paper.title or "Unknown Title"
            venue = paper.venue or ""
            year = getattr(paper, "year", "") or ""
            doi = paper.doi or ""
            # Basic IEEE-style line (keep simple & consistent)
            ref_line = f"[{display_index}] {authors}, \"{title},\" {venue}, {year}. DOI: {doi}"
            refs_from_db.append(ref_line)

        # Only overwrite references if we found anything from DB; otherwise leave existing sections["references"]
        if refs_from_db:
            sections["references"] = "\n".join(refs_from_db)
            print(f"[References Assembly] Built {len(refs_from_db)} references from DB for run {run.id}")
        else:
            # if there is no references created by LLM or DB, ensure there is a fallback message
            if not sections.get("references"):
                sections["references"] = "No references available."
                print(f"[References Assembly] No references found for run {run.id}")
    except Exception as e:
        print(f"[References Assembly] failed to build references from DB: {e}")

    # If user requested a subset of sections and explicitly omitted 'references', drop it from final content
    if requested_sections is not None and "references" not in requested_sections:
        sections.pop("references", None)

    # Step 6: Render DOCX to outputs
    file_id = str(uuid.uuid4())
    out_path = os.path.join(OUTPUT_DIR, f"{file_id}.docx")
    try:
        # Always use the generated title if present. If not, use the uploaded filename
        # but strip the UUID prefix we add at upload time (uuid_origname -> origname).
        if sections.get("title"):
            title_for_doc = sections.get("title")
        else:
            raw_name = os.path.splitext(os.path.basename(run.input_file))[0]
            # if there's an underscore, remove the leading token (commonly the uuid we prefix on upload)
            if "_" in raw_name:
                # remove only one leading token and keep the rest (preserves original filename)
                title_for_doc = raw_name.split("_", 1)[1]
            else:
                title_for_doc = raw_name

        print(f"[File] Rendering DOCX with title: {title_for_doc!r} -> {out_path}")
        render_docx(sections, out_file=out_path, title=title_for_doc)
    except Exception as e:
        print(f"[File Generator] error while rendering DOCX: {e}")
        raise HTTPException(status_code=500, detail="Failed to render document")

    # Step 7: Update DB run status
    crud.update_run_status(db, run.id, status="completed", output_file=out_path)

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
