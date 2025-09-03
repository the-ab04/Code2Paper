from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import tempfile, os, uuid, traceback

from services.code_parser import parse_notebook
from services.llm_generator import generate_sections
from services.citation_manager import enrich_references
from services.file_generator import render_docx  # ✅ Only DOCX generator
from config import MODEL_NAME

router = APIRouter()

@router.post("/generate")
async def generate_paper(
    nb_file: UploadFile = File(...),
    style: str = Form(None),
    title: str = Form(None),
    author: str = Form(None)
):
    try:
        # ✅ Validate file type
        if not nb_file.filename.endswith(".ipynb"):
            raise HTTPException(status_code=400, detail="Only .ipynb files are supported.")

        # ✅ Temporary directory for processing
        tmp_dir = tempfile.mkdtemp(prefix="code2paper_")
        nb_path = os.path.join(tmp_dir, nb_file.filename)
        with open(nb_path, "wb") as f:
            f.write(await nb_file.read())

        # ✅ Parse notebook
        try:
            facts = parse_notebook(nb_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing notebook: {str(e)}")

        # ✅ Generate sections using LLM
        try:
            sections = generate_sections(facts, model_name=MODEL_NAME)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating sections: {str(e)}")

        # ✅ Enrich references with CrossRef
        try:
            sections = enrich_references(sections)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error enriching references: {str(e)}")

        # ✅ Generate DOCX only
        out_file = os.path.join(tmp_dir, f"paper_{uuid.uuid4().hex}.docx")
        try:
            render_docx(sections, out_file, title=title, author=author, style=style)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating DOCX: {str(e)}")

        return FileResponse(out_file, filename="code2paper_output.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unhandled Exception:\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})
