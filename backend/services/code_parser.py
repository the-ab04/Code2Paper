# backend/services/code_parser.py

import ast
import nbformat
from nbformat.reader import NotJSONError
import re
from typing import Dict, List, Optional, Tuple, Any
import importlib
import os
import base64
import uuid
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Configurable storage root and figures dir
STORAGE_ROOT = os.getenv("STORAGE_DIR", "storage")
FIGURES_DIR = Path(STORAGE_ROOT) / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# How many figures to keep (tunable via env)
try:
    DEFAULT_MAX_FIGURES = int(os.getenv("MAX_FIGURES", "3"))
except Exception:
    DEFAULT_MAX_FIGURES = 3

# ---------- AST helpers (unchanged) ----------

def _safe_name(node):
    """Return a readable dotted name for ast nodes (e.g., ast.Attribute chain)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _safe_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _safe_name(node.func)
    return None


def parse_code_ast(source: str) -> Dict[str, Any]:
    """
    Parse Python source code (string) with ast and extract:
      - classes (names, base classes)
      - functions (names)
      - simple top-level assignments for common hyperparameters
      - model usage patterns (torch.nn, tf.keras.layers, Sequential, etc.)
      - dataset loading calls (torchvision.datasets, tf.keras.datasets, pandas.read_csv, etc.)
      - frameworks hints (PyTorch/TensorFlow/scikit-learn)
    """
    out = {
        "classes": [],            # [{"name":..., "bases":[...]}]
        "functions": [],          # ["fn_name", ...]
        "assignments": {},        # var -> simple value or "<complex>"
        "model_defs": [],         # model constructor names or subclass names
        "dataset_calls": [],      # call sites that look like dataset loaders
        "frameworks_found": set(),
    }

    try:
        tree = ast.parse(source)
    except Exception:
        return out

    for node in ast.walk(tree):
        # Classes
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bname = _safe_name(b) or getattr(b, "id", None)
                    if bname:
                        bases.append(bname)
                except Exception:
                    pass
            out["classes"].append({"name": node.name, "bases": bases})

            # Model subclass hint
            for b in bases:
                if re.search(r"nn\.Module|Module|tensorflow\.keras|keras\.Model|Model", str(b)):
                    out["model_defs"].append(node.name)

        # Functions
        if isinstance(node, ast.FunctionDef):
            out["functions"].append(node.name)

        # Top-level assignments (simple constants)
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            var = targets[0]
            val = None
            if isinstance(node.value, ast.Constant):
                val = node.value.value
            elif isinstance(node.value, ast.Num):  # py<3.8 compatibility
                val = node.value.n
            elif isinstance(node.value, ast.Call):
                func_name = _safe_name(node.value.func)
                if func_name and func_name.lower() in ("int", "float", "str"):
                    if node.value.args and isinstance(node.value.args[0], ast.Constant):
                        val = node.value.args[0].value
            if val is not None:
                out["assignments"][var] = val
            else:
                # capture common hyperparam names even if value is complex
                if re.search(r"learning_rate|lr|batch_size|epochs|num_epochs|dropout", var, re.I):
                    out["assignments"][var] = "<complex>"

        # Detect calls (dataset loaders, layer constructors)
        if isinstance(node, ast.Call):
            func_name = _safe_name(node.func)
            if not func_name:
                continue
            fname = func_name.lower()
            # dataset loader heuristics
            if any(k in fname for k in ["datasets", "cifar", "mnist", "imagenet", "read_csv", "tfds", "load_data", "dataset"]):
                out["dataset_calls"].append(func_name)
            # layer / model usage
            if "torch.nn" in fname or "nn." in fname or "keras.layers" in fname or "tf.keras" in fname:
                out["model_defs"].append(func_name)
            # frameworks normalization
            if fname.startswith("torch") or fname.startswith("tf") or "keras" in fname or "sklearn" in fname:
                if "torch" in fname:
                    out["frameworks_found"].add("PyTorch")
                if "keras" in fname or "tf.keras" in fname or fname.startswith("tf"):
                    out["frameworks_found"].add("TensorFlow/Keras")
                if "sklearn" in fname:
                    out["frameworks_found"].add("scikit-learn")

    # tidy outputs
    out["frameworks_found"] = list(out["frameworks_found"])
    out["model_defs"] = list(dict.fromkeys(out["model_defs"]))
    out["dataset_calls"] = list(dict.fromkeys(out["dataset_calls"]))
    out["functions"] = list(dict.fromkeys(out["functions"]))
    return out


# ---------- LLM summarization helper (late import) ----------

def generate_summary_with_llm(concise_context: str, fallback: str = "") -> str:
    """
    Perform a late import of the project's LLM helper to generate a concise (1-3 sentences)
    summary from the AST-based concise_context. If LLM invocation fails, return `fallback`.
    """
    prompt = (
        "You are an assistant that summarizes a machine learning codebase or notebook "
        "in 2–3 concise sentences. Focus on: task/problem, main model/approach, datasets used (if any), "
        "and primary evaluation metrics if evident. Produce only plain text (2–3 sentences).\n\n"
        "Context (extracted from code structure):\n"
        f"{concise_context}\n\n"
        "Return 2–3 concise sentences with no bullet points or JSON."
    )

    try:
        mod = importlib.import_module("services.llm_generator")
        _get_llm = getattr(mod, "_get_llm", None)
        _safe_invoke = getattr(mod, "_safe_invoke", None)
        if _get_llm and _safe_invoke:
            llm = _get_llm()
            out = _safe_invoke(llm, prompt)
            if out and isinstance(out, str) and out.strip():
                # Keep only first 2-3 sentences
                s = out.strip()
                # split into sentences naively
                parts = re.split(r'(?<=[.!?])\s+', s)
                summary = " ".join(parts[:3]).strip()
                if summary:
                    return summary
    except Exception as e:
        logger.debug("LLM summary generation unavailable or failed: %s", e)

    # deterministic fallback: return provided fallback (if any) or a truncated context
    if fallback:
        return fallback
    text = re.sub(r"\s+", " ", concise_context).strip()
    if not text:
        return "A machine learning codebase (no clear summary)."
    # return up to ~180 chars to keep it query-friendly
    return (text[:180] + "...") if len(text) > 180 else text


# ---------- Helpers for outputs / metrics extraction ----------

_METRIC_REGEX = re.compile(
    r"\b(?P<metric>accuracy|acc|f1[-_ ]?score|precision|recall|auc|loss|mse|rmse)\b\s*[:=]?\s*(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    flags=re.I,
)


def _extract_metrics_from_text(text: str) -> Dict[str, float]:
    """
    Return a mapping metric_name -> numeric value for all metric-like occurrences in the provided text.
    Lowercase metric names normalized to standard tokens (accuracy, loss, f1, precision, recall, auc, mse, rmse).
    """
    metrics_found: Dict[str, float] = {}
    if not text:
        return metrics_found
    for m in _METRIC_REGEX.finditer(text):
        key = m.group("metric") or ""
        val = m.group("value") or ""
        try:
            num = float(val)
            # normalize metric key
            k = key.lower().replace(" ", "_").replace("-", "_")
            if k in ("acc",):
                k = "accuracy"
            if k.startswith("f1"):
                k = "f1"
            metrics_found[k] = num
        except Exception:
            continue
    return metrics_found


def _save_base64_image(raw: Any, mime: str) -> Optional[str]:
    """
    Decode base64-like raw data and save to FIGURES_DIR returning the filepath string.
    raw may be a base64 string, a list of strings, or bytes.
    """
    if raw is None:
        return None
    # flatten list -> str
    if isinstance(raw, list):
        raw = "".join(raw)
    # If data URI like "data:image/png;base64,.."
    if isinstance(raw, str) and raw.startswith("data:"):
        parts = raw.split("base64,", 1)
        if len(parts) == 2:
            raw = parts[1]
    bin_data = None
    try:
        if isinstance(raw, str):
            # strip whitespace which sometimes breaks decoding
            raw_str = raw.strip()
            bin_data = base64.b64decode(raw_str, validate=False)
        elif isinstance(raw, (bytes, bytearray)):
            bin_data = bytes(raw)
        else:
            return None
    except Exception:
        # sometimes raw is not base64 but already binary in strange form; try to coerce
        try:
            if isinstance(raw, (bytes, bytearray)):
                bin_data = bytes(raw)
        except Exception:
            return None

    if not bin_data:
        return None

    # choose ext from mime
    ext = "png"
    if "jpeg" in (mime or "") or "jpg" in (mime or ""):
        ext = "jpg"
    if "svg" in (mime or ""):
        ext = "svg"
    fname = f"{uuid.uuid4().hex}.{ext}"
    p = FIGURES_DIR / fname
    try:
        with open(p, "wb") as fh:
            fh.write(bin_data)
        return str(p)
    except Exception as e:
        logger.debug("Failed to save image to disk: %s", e)
        return None


def _extract_caption_hint_from_code_block(code_block: str) -> Optional[str]:
    """
    Look for plt.title/plt.xlabel/plt.ylabel/ax.set_title/ax.set_xlabel/ax.set_ylabel, sns.title, etc.
    Return a short cleaned hint (first match).
    """
    if not code_block:
        return None
    # look for common plotting calls with string args
    patterns = [
        r"plt\.title\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"plt\.xlabel\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"plt\.ylabel\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"ax\.set_title\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"ax\.set_xlabel\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"ax\.set_ylabel\s*\(\s*['\"]([^'\"]{2,200})['\"]\s*\)",
        r"sns\.heatmap\s*\(",
        r"plt\.imshow\s*\(",
        r"plt\.plot\s*\(",
    ]
    for pat in patterns:
        m = re.search(pat, code_block, flags=re.I)
        if m:
            # if group captured text, return it; otherwise return function name hint
            if m.groups():
                hint = m.group(1)
                if hint:
                    return hint.strip()[:240]
            # fallback hints
            name_hint = pat.split("\\")[0].split(".")[0]
            return name_hint
    return None


def _select_useful_figures(images: List[Dict[str, Any]], outputs: List[Dict[str, Any]], code_blocks: List[str], max_figs: int = DEFAULT_MAX_FIGURES) -> List[Dict[str, Any]]:
    """
    Heuristic scoring to pick the most useful figures:
      - preference if nearby cell output contains metric keywords (accuracy, loss, etc.)
      - slight preference for png/jpeg and larger file size
      - prefer figures appearing later (assumes later figures are results)
      - attempt to attach a caption_hint from code (xlabel/title) or from output text
    Returns a subset (<= max_figs) preserving metadata and a short caption_hint.
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for img in images:
        score = 0.0
        path = img.get("path")
        mime = img.get("mime", "")
        cell_idx = img.get("cell_index", None)

        # file-size bump
        try:
            sz = os.path.getsize(path)
            score += min(3.0, sz / 100_000.0)
        except Exception:
            pass

        # mime preference
        if "png" in (mime or "") or "jpeg" in (mime or "") or "jpg" in (mime or ""):
            score += 0.5

        # prefer images from code cells that had metric outputs
        hint = ""
        if isinstance(outputs, list) and cell_idx is not None:
            for o in outputs:
                if o.get("cell_index") == cell_idx:
                    txt = (o.get("text") or "") + " " + (o.get("error") or "")
                    if re.search(r"\b(accuracy|acc|loss|f1|precision|recall|auc|mse|rmse)\b", txt, flags=re.I):
                        score += 3.0
                    # use first non-empty line from output text as hint
                    first_line = ""
                    for ln in (txt or "").splitlines():
                        ln2 = ln.strip()
                        if ln2:
                            first_line = ln2
                            break
                    if first_line:
                        hint = first_line[:240]
                    break

        # try to extract caption hint from corresponding code block if available
        if (not hint) and isinstance(code_blocks, list) and cell_idx is not None and 0 <= cell_idx < len(code_blocks):
            code_hint = _extract_caption_hint_from_code_block(code_blocks[cell_idx])
            if code_hint:
                hint = code_hint
                score += 1.5

        # prefer later images (results plots are often later)
        if cell_idx is not None:
            score += (cell_idx or 0) * 0.01

        # Provide a fallback hint if none found: use filename
        if not hint:
            hint = os.path.basename(path)

        scored.append((score, {"path": path, "mime": mime, "cell_index": cell_idx, "caption_hint": hint}))

    # sort and take top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [entry for _, entry in scored[:max_figs]]
    return selected


# ---------- Main: parse_notebook (AST + LLM summary + outputs/metrics) ----------

def parse_notebook(nb_path: str, max_figures: int = DEFAULT_MAX_FIGURES) -> Dict[str, Any]:
    """
    Parse an .ipynb or .py file and extract AST-based facts, outputs (including images),
    and produce a short LLM-driven summary.

    Returned dict includes:
      - summary: str (1–3 sentence summary generated by LLM or fallback)
      - code: List[str] code cell texts (or single element for .py)
      - model_defs: List[str] model constructor usages or Model subclass names
      - dataset_calls: List[str] detected dataset loader call sites
      - hyperparameters: Dict[str, Any] simple top-level assignments (lr, batch_size, ...)
      - classes: List[Dict] class defs (name, bases)
      - functions: List[str] function names
      - frameworks: List[str] frameworks detected (PyTorch/TensorFlow/scikit-learn)
      - outputs: List[Dict] of per-code-cell outputs: {cell_index, text, error, images}
      - images: List[Dict] of saved image records {path, mime, cell_index}
      - figures: List[Dict] selected useful figures {path, mime, cell_index, caption_hint}
      - final_metrics / output_metrics: Dict[str, float] mapping metric -> last observed numeric value
      - combined_text: small snippet for backward compatibility
    """
    # support .py fallback: treat whole file as a code block
    ext = os.path.splitext(nb_path)[1].lower()
    md_blocks: List[str] = []
    code_blocks: List[str] = []
    outputs_list: List[Dict[str, Any]] = []
    images_list: List[Dict[str, Any]] = []
    collected_metric_occurrences: List[Tuple[str, float, int]] = []  # (metric, value, order)

    global_metric_counter = 0

    if ext == ".py":
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                src = f.read()
            code_blocks = [src]
        except Exception as e:
            raise ValueError(f"Error reading python script: {e}")
    else:
        # attempt to read as notebook; if that fails try to read as plain text
        try:
            nb = nbformat.read(nb_path, as_version=4)
            for raw_cell_index, cell in enumerate(nb.cells):
                cell_type = cell.get("cell_type")
                source = cell.get("source", "") or ""
                if isinstance(source, list):
                    source = "\n".join(source)
                source = source.strip()

                if cell_type == "code":
                    # append code block first so code_blocks index matches cell index
                    code_blocks.append(source)

                    # handle outputs for this code cell
                    cell_outputs = cell.get("outputs", []) or []
                    text_parts = []
                    img_paths_for_cell: List[str] = []
                    error_text = ""
                    for out in cell_outputs:
                        out_type = out.get("output_type", "")
                        # stream output (stdout / stderr)
                        if out_type == "stream":
                            txt = out.get("text", "") or out.get("name", "")
                            if isinstance(txt, list):
                                txt = "".join(txt)
                            if txt:
                                text_parts.append(str(txt))
                                metrics_here = _extract_metrics_from_text(str(txt))
                                for mk, mv in metrics_here.items():
                                    global_metric_counter += 1
                                    collected_metric_occurrences.append((mk, mv, global_metric_counter))
                        # display_data / execute_result may include image mime or text/plain
                        if out_type in ("display_data", "execute_result"):
                            data = out.get("data", {}) or {}
                            # extract textual representation if present
                            txt = data.get("text/plain")
                            if txt:
                                if isinstance(txt, list):
                                    txt = "".join(txt)
                                text_parts.append(str(txt))
                                metrics_here = _extract_metrics_from_text(str(txt))
                                for mk, mv in metrics_here.items():
                                    global_metric_counter += 1
                                    collected_metric_occurrences.append((mk, mv, global_metric_counter))
                            # check for common image mimes
                            for mime in ("image/png", "image/jpeg", "image/jpg", "image/svg+xml"):
                                if mime in data:
                                    raw = data.get(mime)
                                    saved = _save_base64_image(raw, mime)
                                    if saved:
                                        img_paths_for_cell.append(saved)
                                        images_list.append({"path": saved, "mime": mime, "cell_index": len(code_blocks) - 1})
                        # plain text in output
                        if out_type == "error":
                            ename = out.get("ename", "")
                            evalue = out.get("evalue", "")
                            tb = out.get("traceback", []) or []
                            tbtext = "\n".join(tb) if isinstance(tb, list) else str(tb)
                            err = f"{ename}: {evalue}\n{tbtext}"
                            error_text += err
                            metrics_here = _extract_metrics_from_text(err)
                            for mk, mv in metrics_here.items():
                                global_metric_counter += 1
                                collected_metric_occurrences.append((mk, mv, global_metric_counter))
                        # sometimes outputs have 'text' directly
                        if isinstance(out, dict) and "text" in out and out.get("text"):
                            txt = out.get("text")
                            if isinstance(txt, list):
                                txt = "".join(txt)
                            text_parts.append(str(txt))
                            metrics_here = _extract_metrics_from_text(str(txt))
                            for mk, mv in metrics_here.items():
                                global_metric_counter += 1
                                collected_metric_occurrences.append((mk, mv, global_metric_counter))

                    cell_text = "\n".join([p for p in text_parts if p]).strip()
                    outputs_list.append({
                        "cell_index": len(code_blocks) - 1,
                        "text": cell_text,
                        "error": error_text.strip() or None,
                        "images": img_paths_for_cell,
                    })

                elif cell_type == "markdown":
                    if source:
                        md_blocks.append(source)

        except NotJSONError:
            # fallback: try reading file as plain python
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    src = f.read()
                code_blocks = [src]
            except Exception as e:
                raise ValueError(f"Error reading notebook/script: {e}")
        except Exception as e:
            raise ValueError(f"Error reading notebook: {e}")

    # perform AST analysis on each code block and merge results
    ast_aggregate = {
        "classes": [],
        "functions": [],
        "assignments": {},
        "model_defs": [],
        "dataset_calls": [],
        "frameworks_found": []
    }

    for code in code_blocks:
        try:
            info = parse_code_ast(code)
            ast_aggregate["classes"].extend(info.get("classes", []))
            ast_aggregate["functions"].extend(info.get("functions", []))
            ast_aggregate["model_defs"].extend(info.get("model_defs", []))
            ast_aggregate["dataset_calls"].extend(info.get("dataset_calls", []))
            ast_aggregate["frameworks_found"].extend(info.get("frameworks_found", []))
            for k, v in info.get("assignments", {}).items():
                # prefer simple values, but keep first-seen
                ast_aggregate["assignments"].setdefault(k, v)
        except Exception:
            # ignore AST errors per cell
            continue

    # de-dupe lists while preserving order
    def _uniq(seq):
        seen = set()
        out = []
        for s in seq:
            # if elements are dicts (classes), use a representative key
            key = s
            if isinstance(s, dict):
                key = s.get("name")
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    ast_aggregate["model_defs"] = _uniq(ast_aggregate["model_defs"])
    ast_aggregate["dataset_calls"] = _uniq(ast_aggregate["dataset_calls"])
    ast_aggregate["functions"] = _uniq(ast_aggregate["functions"])
    ast_aggregate["frameworks_found"] = _uniq(ast_aggregate["frameworks_found"])
    ast_aggregate["classes"] = _uniq(ast_aggregate["classes"])

    # Build a concise_context for LLM summarization using only AST-derived facts + extracted metric summary
    concise_parts = []
    # prefer model_defs and class names (they directly indicate models used)
    if ast_aggregate["model_defs"]:
        concise_parts.append("Model hints: " + ", ".join(ast_aggregate["model_defs"][:6]))
    if ast_aggregate["classes"]:
        concise_parts.append("Classes: " + ", ".join(c.get("name") for c in ast_aggregate["classes"][:6]))
    if ast_aggregate["dataset_calls"]:
        concise_parts.append("Dataset calls: " + ", ".join(ast_aggregate["dataset_calls"][:6]))
    if ast_aggregate["frameworks_found"]:
        concise_parts.append("Frameworks: " + ", ".join(ast_aggregate["frameworks_found"]))
    # short hyperparam summary (top common keys)
    hp_keys = []
    for k in ("learning_rate", "lr", "batch_size", "epochs", "num_epochs", "dropout"):
        if k in ast_aggregate["assignments"]:
            hp_keys.append(f"{k}={ast_aggregate['assignments'][k]}")
    if hp_keys:
        concise_parts.append("Hyperparams: " + ", ".join(hp_keys))

    # include a short code preview (first non-empty code block trimmed)
    code_preview = ""
    for c in code_blocks:
        if c and isinstance(c, str) and c.strip():
            snippet = c.strip().splitlines()
            preview_lines = []
            for ln in snippet[:6]:
                ln2 = ln.strip()
                if ln2:
                    preview_lines.append(ln2)
            if preview_lines:
                code_preview = "Code preview: " + " | ".join(preview_lines[:6])
            break
    if code_preview:
        concise_parts.append(code_preview)

    # summarize metric occurrences (take last-occurrence per metric name)
    final_metrics: Dict[str, float] = {}
    if collected_metric_occurrences:
        # keep last seen numeric value for each metric
        for mk, mv, order in collected_metric_occurrences:
            final_metrics[mk] = mv
        # add a short metrics summary string
        metric_summary = ", ".join(f"{k}={v}" for k, v in final_metrics.items())
        if metric_summary:
            concise_parts.append("Observed metrics: " + metric_summary)

    concise_context = "\n".join(concise_parts).strip() or (code_preview or "No AST-derived context available.")

    # Fallback short summary if LLM cannot be used
    fallback_summary = "A machine learning codebase."
    if ast_aggregate["model_defs"]:
        fallback_summary = "Code uses " + ", ".join(ast_aggregate["model_defs"][:3]) + "."
    elif ast_aggregate["classes"]:
        fallback_summary = "Contains classes: " + ", ".join(c.get("name") for c in ast_aggregate["classes"][:3]) + "."

    # Generate summary using LLM (late import). This is the only place we contact the LLM.
    # We include the concise_context (which contains metric summary when available) so the LLM output
    # will incorporate final metric values into the summary and thus this summary can be used as a better query.
    summary = generate_summary_with_llm(concise_context=concise_context, fallback=fallback_summary)
    summary = summary.strip()

    # select useful figures from images_list
    selected_figures = _select_useful_figures(images_list, outputs_list, code_blocks, max_figs=max_figures) if images_list else []

    # Build returned facts dict (keeps minimal fields but includes AST outputs + outputs/images/metrics)
    facts: Dict[str, Any] = {
        "summary": summary,
        "code": code_blocks,
        "model_defs": ast_aggregate["model_defs"],
        "dataset_calls": ast_aggregate["dataset_calls"],
        "hyperparameters": ast_aggregate["assignments"],
        "classes": ast_aggregate["classes"],
        "functions": ast_aggregate["functions"],
        "frameworks": ast_aggregate["frameworks_found"],
        "outputs": outputs_list,
        "images": images_list,            # raw saved images
        "figures": selected_figures,      # selected useful figures (top-N) with caption_hint
        # Keep both names for compatibility:
        "final_metrics": final_metrics,
        "output_metrics": final_metrics,  # alias preferred by other modules
        # keep a small combined_text for backward compatibility (limited)
        "combined_text": (concise_context or code_preview)[:1200],
    }

    # also populate metrics list in a compact form for backwards compatibility
    metric_list = []
    for k, v in final_metrics.items():
        metric_list.append(f"{k}: {v}")
    if metric_list:
        facts.setdefault("metrics", []).extend(metric_list)

    return facts


# quick test guard (unchanged)
if __name__ == "__main__":
    test_path = "sample_inputs/sample_notebook.ipynb"
    try:
        facts = parse_notebook(test_path)
        print("Notebook facts extracted:")
        for key, value in facts.items():
            if isinstance(value, list):
                preview = value[:5]
                print(f"{key} ({len(value)}): {preview} ...")
            else:
                print(f"{key}: {str(value)[:200]}")
    except Exception as e:
        print(f"Error: {e}")
