# backend/services/pdf_generator.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import datetime
import os

def build_pdf_from_sections(sections_dict, metadata, citation, output_path):
    """
    Create a readable PDF using ReportLab.
    sections_dict: {"abstract": "...", "methodology": "..."}
    metadata: dict with parsed info
    citation: dict or None
    output_path: path to write the PDF
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = inch
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    title = f"Generated Paper — {datetime.date.today().isoformat()}"
    c.drawString(margin, y, title)
    y -= 0.4 * inch

    # Metadata block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Extracted Metadata:")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for k, v in metadata.get("simple_meta", {}).items():
        line = f"{k}: {v}"
        y = _draw_wrapped_line(c, line, margin, y, width - 2 * margin)
        y -= 0.05 * inch
        if y < margin:
            c.showPage(); y = height - margin

    y -= 0.15 * inch

    # Sections
    for name, text in sections_dict.items():
        c.setFont("Helvetica-Bold", 13)
        y = _draw_wrapped_line(c, name.capitalize(), margin, y, width - 2 * margin)
        y -= 0.12 * inch
        c.setFont("Helvetica", 10)
        y = _draw_wrapped_paragraph(c, text, margin, y, width - 2 * margin)
        y -= 0.2 * inch
        if y < margin:
            c.showPage(); y = height - margin

    # References
    if citation:
        c.setFont("Helvetica-Bold", 12)
        y = _draw_wrapped_line(c, "References", margin, y, width - 2 * margin)
        y -= 0.12 * inch
        c.setFont("Helvetica", 10)
        auth = ", ".join(citation.get("authors", [])[:3])
        ref_line = f"{auth} ({citation.get('year')}). {citation.get('title')}. DOI: {citation.get('doi')}"
        y = _draw_wrapped_paragraph(c, ref_line, margin, y, width - 2 * margin)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c.save()

def _draw_wrapped_line(c, text, x, y, max_width):
    # simple writer: if text would overflow horizontally, we place at new line
    if y < inch:
        c.showPage()
        y = A4[1] - inch
    c.drawString(x, y, text)
    return y - 14

def _draw_wrapped_paragraph(c, text, x, y, max_width, line_height=12):
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            # draw current line
            if y < inch:
                c.showPage(); y = A4[1] - inch
            c.drawString(x, y, line)
            y -= line_height
            line = w
    if line:
        if y < inch:
            c.showPage(); y = A4[1] - inch
        c.drawString(x, y, line)
        y -= line_height
    return y
