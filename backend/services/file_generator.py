from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from typing import Dict

def render_docx(sections: Dict[str, str], out_file: str, title: str = None, author: str = None, style: str = None):
    """
    Generate a formatted DOCX research paper.
    sections: dict with keys like 'abstract', 'introduction', etc.
    """

    doc = Document()

    # ✅ Title Page
    if title:
        title_para = doc.add_paragraph(title)
        title_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        title_run = title_para.runs[0]
        title_run.bold = True
        title_run.font.size = Pt(22)

    if author:
        author_para = doc.add_paragraph(f"Author: {author}")
        author_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        author_para.runs[0].font.size = Pt(14)

    doc.add_paragraph()  # Empty line
    doc.add_page_break()

    # ✅ Sections in proper order
    section_order = [
        "abstract", "introduction", "methods", "experiments",
        "results", "discussion", "conclusion", "references"
    ]

    for section_name in section_order:
        content = sections.get(section_name, "").strip()
        if not content:
            continue

        # Section heading
        heading = section_name.capitalize()
        doc.add_heading(heading, level=1)

        # Content formatting
        if section_name == "references":
            # Add references as numbered list
            for line in content.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip(), style="List Number")
        else:
            p = doc.add_paragraph(content)
            p.paragraph_format.space_after = Pt(12)

        doc.add_paragraph()  # Add extra space after section

    # ✅ Save DOCX
    doc.save(out_file)
