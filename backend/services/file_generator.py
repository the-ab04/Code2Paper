# backend/services/file_generator.py

import io
from typing import Dict, Union
from docx import Document


def render_docx(
    sections: Dict[str, str],
    out_file: Union[str, io.BytesIO],
    title: str = "Generated Paper"
) -> None:
    """
    Render a DOCX file from structured paper sections.
    Can write to a file path (str) or to a BytesIO buffer.

    Args:
        sections: dict with keys like "abstract", "methods", "results", etc.
        out_file: file path (str) to save, or BytesIO buffer for streaming.
        title: document title (paper title).
    """
    doc = Document()

    # === Title ===
    if title.strip():
        doc.add_heading(title.strip(), level=0)

    # === Main Sections ===
    for key in [
        "abstract", "introduction", "methods",
        "experiments", "results", "discussion", "conclusion"
    ]:
        content = sections.get(key, "").strip()
        if content:
            doc.add_heading(key.capitalize(), level=1)
            doc.add_paragraph(content)

    # === References ===
    refs = sections.get("references", "").strip()
    if refs:
        doc.add_heading("References", level=1)
        for ref in refs.split("\n"):
            if ref.strip():
                doc.add_paragraph(ref.strip(), style="List Number")

    # === Save ===
    if isinstance(out_file, str):
        # Save to disk
        doc.save(out_file)
    elif isinstance(out_file, io.BytesIO):
        # Save to memory buffer (for StreamingResponse in FastAPI)
        doc.save(out_file)
        out_file.seek(0)
    else:
        raise TypeError("out_file must be a str (path) or io.BytesIO")
