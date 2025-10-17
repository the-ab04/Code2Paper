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
    Handles missing styles safely and ensures references are properly formatted.
    """
    doc = Document()

    # === Title ===
    if title and title.strip():
        doc.add_heading(title.strip(), level=0)

    # === Main Sections ===
    ordered_sections = [
        "abstract", "introduction", "methods",
        "experiments", "results", "discussion", "conclusion"
    ]

    for key in ordered_sections:
        content = sections.get(key, "")
        if not content or not content.strip():
            continue

        # Add section heading
        doc.add_heading(key.capitalize(), level=1)

        # Split content by double newlines into paragraphs
        for para in [p.strip() for p in content.split("\n\n") if p.strip()]:
            doc.add_paragraph(para)

    # === References ===
    refs = sections.get("references", "")
    if refs is not None:
        refs = refs.strip()

    # Always create a References section
    doc.add_page_break()
    doc.add_heading("References", level=1)

    if refs:
        for line in refs.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                # Try using numbered list style if available
                doc.add_paragraph(line, style="List Number")
            except KeyError:
                # Fallback if the style doesn't exist in the template
                doc.add_paragraph(line)
    else:
        doc.add_paragraph("No references available.")

    # === Save ===
    try:
        if isinstance(out_file, str):
            doc.save(out_file)
        elif isinstance(out_file, io.BytesIO):
            doc.save(out_file)
            out_file.seek(0)
        else:
            raise TypeError("out_file must be a str (path) or io.BytesIO")
    except Exception as e:
        # Explicitly raise errors for debugging in backend logs
        raise RuntimeError(f"Error while saving DOCX: {e}")
