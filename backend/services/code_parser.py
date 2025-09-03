import nbformat
from nbformat.reader import NotJSONError
import re
from typing import Dict, List

def parse_notebook(nb_path: str) -> Dict[str, List[str]]:
    """
    Extract markdown text, code blocks, and training logs from a Jupyter notebook (.ipynb).
    
    Args:
        nb_path (str): Path to the .ipynb file.
    
    Returns:
        dict: {
            "markdown": [list of markdown text],
            "code": [list of code blocks],
            "logs": [list of detected training logs or metrics]
        }
    
    Raises:
        ValueError: If the notebook file is invalid or cannot be parsed.
    """
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except NotJSONError as e:
        raise ValueError(f"Invalid notebook format: {e}")
    except Exception as e:
        raise ValueError(f"Error reading notebook: {e}")

    md_blocks: List[str] = []
    code_blocks: List[str] = []
    logs: List[str] = []

    for cell in nb.cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", "").strip()

        if not source:
            continue  # Skip empty cells

        if cell_type == "markdown":
            md_blocks.append(source)

        elif cell_type == "code":
            code_blocks.append(source)

            # Detect metrics or training logs (accuracy, loss, f1, precision, recall)
            for line in source.splitlines():
                if re.search(r"(accuracy|loss|f1|precision|recall)\s*[:=]\s*\d+(\.\d+)?", line, re.I):
                    logs.append(line.strip())

    return {
        "markdown": md_blocks,
        "code": code_blocks,
        "logs": logs
    }
