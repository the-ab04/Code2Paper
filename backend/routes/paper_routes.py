# backend/routes/paper_routes.py

import os
import uuid
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from services.code_parser import parse_code_metadata
from services.llm_generator import generate_section_text, init_llm
from services.citation_manager import find_top_crossref_for_term
from services.pdf_generator import build_pdf_from_sections

# Create blueprint
paper_bp = Blueprint("paper_bp", __name__)

# Directories (relative to backend/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track whether LLM init has been done
_first_init_done = False


@paper_bp.before_app_request
def startup():
    """
    Initializes heavy resources (like LLM) once, before handling the first request.
    Flask 3.x removed before_app_first_request for blueprints, so we handle manually.
    """
    global _first_init_done
    if not _first_init_done:
        current_app.logger.info("Initializing LLM for the first request...")
        init_llm()
        _first_init_done = True


@paper_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Accepts multipart form file field named 'file'.
    Returns JSON with saved file_path on success.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    uid = str(uuid.uuid4())[:8]
    save_name = f"{uid}_{filename}"
    path = os.path.join(UPLOAD_DIR, save_name)
    file.save(path)
    return jsonify({"status": "uploaded", "file_path": path})


@paper_bp.route("/generate", methods=["POST"])
def generate():
    """
    POST JSON:
    {
      "file_path": "backend/uploads/abcd_sample.py"  (or relative path),
      "sections": ["abstract","methodology","results"]
    }
    """
    data = request.get_json(force=True)
    file_path = data.get("file_path")
    sections = data.get("sections", ["abstract", "methodology", "results", "conclusion"])

    if not file_path:
        return jsonify({"error": "file_path field is required"}), 400

    # Allow either absolute path or path relative to backend
    if not os.path.isabs(file_path):
        file_path = os.path.join(BASE_DIR, file_path)
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        return jsonify({"error": "file not found", "path": file_path}), 400

    # 1) Parse code and extract metadata
    metadata = parse_code_metadata(file_path)

    # 2) Generate the requested sections
    results = {}
    for sec in sections:
        prompt_data = {"section": sec, "metadata": metadata}
        text = generate_section_text(prompt_data)
        results[sec] = text

    # 3) Get a citation candidate (non-blocking)
    citation = None
    try:
        keywords = metadata.get("keywords", [])
        if keywords:
            citation = find_top_crossref_for_term(keywords[0])
    except Exception:
        citation = None

    # 4) Create PDF
    pdf_filename = f"{str(uuid.uuid4())[:8]}_paper.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    build_pdf_from_sections(results, metadata, citation, pdf_path)

    return jsonify({
        "status": "generated",
        "sections": list(results.keys()),
        "pdf_path": pdf_path,
        "citation": citation
    })


@paper_bp.route("/download", methods=["GET"])
def download():
    """
    Query param: ?path=/absolute/or/relative/path/to/pdf
    """
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "path param required"}), 400

    # If relative, assume outputs folder
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
        path = os.path.abspath(path)

    if not os.path.exists(path):
        return jsonify({"error": "file not found", "path": path}), 404

    return send_file(path, as_attachment=True)
