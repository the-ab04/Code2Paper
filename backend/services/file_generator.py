import io
import os
import re
from typing import Dict, Union
from docx import Document


def render_docx(
    sections: Dict[str, str],
    out_file: Union[str, io.BytesIO],
    title: str = "Generated Paper"
) -> None:
    """
    Render a DOCX file from structured paper sections.

    Improvements:
      - Treats 'literature_review' as "Literature Review".
      - Splits regular sections into paragraphs on blank lines; preserves simple bullets.
      - For References:
          * Collapses internal newlines/extra whitespace inside each reference.
          * Inserts a newline before every bracketed index like "[1]" so each reference starts on its own line.
          * Writes each reference as a single paragraph (no DOCX numbered list style).
    """
    doc = Document()

    # === Title ===
    if title and title.strip():
        doc.add_heading(title.strip(), level=0)

    # === Main Sections ===
    ordered_sections = [
        "abstract", "introduction", "literature_review", "methods",
        "experiments", "results", "discussion", "conclusion"
    ]

    def pretty_heading(key: str) -> str:
        """Convert internal key -> user-friendly heading."""
        return key.replace("_", " ").strip().title()

    def split_into_paragraphs(text: str) -> list:
        """
        Split a block of section text into paragraph strings.
        Primary separator: blank line(s).
        Also treat lines that are all bullet-like as separate paragraphs.
        """
        if not text:
            return []
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p and p.strip()]
        paras = []
        for p in raw_paras:
            lines = [l.rstrip() for l in p.split("\n") if l.strip()]
            # If every line looks like a bullet, keep each as a separate paragraph (remove marker)
            bullet_like = sum(1 for l in lines if re.match(r"^\s*[-\*\u2022]\s+", l))
            if bullet_like and bullet_like == len(lines):
                for l in lines:
                    cleaned = re.sub(r"^\s*[-\*\u2022]\s+", "", l).strip()
                    if cleaned:
                        paras.append(f"• {cleaned}")
            else:
                paras.append(p)
        return paras

    for key in ordered_sections:
        content = sections.get(key, "")
        if not content or not content.strip():
            continue

        heading_text = pretty_heading(key)
        doc.add_heading(heading_text, level=1)

        for para in split_into_paragraphs(content):
            if isinstance(para, str) and para.lstrip().startswith("• "):
                doc.add_paragraph(para.strip())
            else:
                cleaned_para = re.sub(r"\s+", " ", para).strip()  # collapse internal whitespace/newlines
                doc.add_paragraph(cleaned_para)

    # === References ===
    refs = sections.get("references", "")
    if refs is not None:
        refs = refs.strip()

    # Always create a References section on a new page
    doc.add_page_break()
    doc.add_heading("References", level=1)

    if refs:
        # Step A: Normalize line endings and trim
        refs_norm = refs.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Step B: Collapse multiple whitespace and internal newlines into single spaces for each potential reference.
        # We will later re-insert separators before bracketed indices.
        # Use a temporary collapse to make sure multi-line references become single-line.
        refs_collapsed = re.sub(r"\s+", " ", refs_norm).strip()

        # Step C: Insert a newline before every bracketed numeric index like "[1]", "[2]", ...
        # We add "\n" before the bracket. This will create a leading newline if the first token is numbered;
        # we strip any leading newline below.
        # The regex captures the number so that we preserve it.
        refs_with_breaks = re.sub(r"\s*\[(\d+)\]\s*", r"\n[\1] ", refs_collapsed)

        # Remove a leading newline if present (so the first reference doesn't start with an empty paragraph)
        refs_with_breaks = refs_with_breaks.lstrip("\n").strip()

        # If the references use other placeholder forms (e.g., [CITATION: ...]) and are not numeric,
        # we also attempt to split on occurrences of '[' that look like a new reference only when followed by a digit.
        # The logic above handles numeric bracketed indices. For other formats you'd adapt accordingly.

        # Step D: Split into reference blocks by newline (each line now should start with a bracket like "[1]")
        ref_lines = [line.strip() for line in refs_with_breaks.split("\n") if line and line.strip()]

        # Step E: Write each reference as a single paragraph
        for line in ref_lines:
            # Optionally remove leading numeric bracket, e.g. to have plain references without "[1]":
            # uncomment the following line:
            # line = re.sub(r'^\s*\[\d+\]\s*', '', line)
            doc.add_paragraph(line)
    else:
        doc.add_paragraph("No references available.")

    # === Save the document ===
    try:
        if isinstance(out_file, str):
            parent = os.path.dirname(out_file)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            doc.save(out_file)
        elif isinstance(out_file, io.BytesIO):
            doc.save(out_file)
            out_file.seek(0)
        else:
            raise TypeError("out_file must be a str (path) or io.BytesIO")
    except Exception as e:
        raise RuntimeError(f"Error while saving DOCX: {e}")
