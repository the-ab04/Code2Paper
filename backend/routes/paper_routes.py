# backend/routes/paper_routes.py

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import shutil, os, uuid

from db.base import get_db
from db import crud, schemas
from services.code_parser import parse_notebook
from services.llm_generator import generate_sections
from services.file_generator import render_docx
from services.citation_manager import enrich_references
from services.paper_finder import find_papers  # external papers
from services.reset_manager import full_reset    # ✅ NEW

router = APIRouter(prefix="/api/paper", tags=["paper"])

UPLOAD_DIR = "storage/uploads"
OUTPUT_DIR = "storage/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Upload Notebook/Code to Start a Run ===
@router.post("/upload", response_model=schemas.RunOut)
def upload_notebook(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a Jupyter Notebook (.ipynb) or script file.
    Resets DB, Qdrant, and storage before creating a new Run.
    """
    # Step 0: Reset everything for a fresh run
    full_reset(db)

    if not (file.filename.endswith(".ipynb") or file.filename.endswith(".py")):
        raise HTTPException(status_code=400, detail="Only .ipynb or .py files are allowed")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    run = crud.create_run(db=db, input_file=save_path)
    return run


# === Generate Paper from a Run ===
@router.post("/generate/{run_id}")
def generate_paper(run_id: int, db: Session = Depends(get_db)):
    """
    Parse notebook, generate sections with LLM,
    discover related papers, enrich with citations,
    render DOCX, save file, and return download URL.
    """
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not run.input_file or not os.path.exists(run.input_file):
        raise HTTPException(status_code=400, detail="Input file is missing")

    # Step 1: Parse notebook → structured facts
    facts = parse_notebook(run.input_file)

    # Step 2: Generate paper sections with LLM
    sections = generate_sections(facts)

    # Step 3: Find external papers (CrossRef + ArXiv + index in Qdrant)
    queries = facts.get("datasets", []) + facts.get("metrics", [])
    if not queries:
        queries = ["machine learning"]  # fallback
    external_papers = find_papers(queries, max_results=5)
    print(f"✅ Found and indexed {len(external_papers)} external papers")

    # Step 4: Enrich references (resolve placeholders, insert papers/citations in DB)
    sections = enrich_references(sections, run_id=run.id, db=db)

    # Step 5: Render DOCX to outputs folder
    file_id = str(uuid.uuid4())
    out_path = os.path.join(OUTPUT_DIR, f"{file_id}.docx")
    render_docx(sections, out_file=out_path, title=sections.get("title", "Generated Paper"))

    # Step 6: Update DB with file path + completed status
    crud.update_run_status(db, run.id, status="completed", output_file=out_path)

    return {
        "run_id": run.id,
        "download_url": f"/api/paper/download/{run.id}"
    }


# === Download Generated Paper ===
@router.get("/download/{run_id}")
def download_paper(run_id: int, db: Session = Depends(get_db)):
    """
    Serve the generated DOCX file for download.
    """
    run = crud.get_run(db, run_id)
    if not run or not run.output_file or not os.path.exists(run.output_file):
        raise HTTPException(status_code=404, detail="Generated file not found")

    filename = os.path.basename(run.output_file)
    return FileResponse(
        path=run.output_file,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


# === Get Run by ID (with linked papers/citations) ===
@router.get("/runs/{run_id}", response_model=schemas.RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = crud.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# === List All Runs ===
@router.get("/runs", response_model=List[schemas.RunOut])
def list_runs(db: Session = Depends(get_db)):
    return db.query(crud.models.Run).all()
