# backend/services/code_parser.py
"""
Lightweight code parser to extract metadata for LLM prompts.
This is intentionally simple — extend with stronger heuristics later.
"""
import ast
import re

def _extract_strings_from_ast(tree):
    strings = []
    for node in ast.walk(tree):
        # Python 3.8+: ast.Constant, older: ast.Str
        if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
        elif isinstance(node, ast.Str):
            strings.append(node.s)
    return strings

def parse_code_metadata(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    # keywords from regex (expand this list as needed)
    keywords = set()
    datasets = ["mnist", "cifar10", "cifar-10", "imagenet", "fashion_mnist", "iris"]
    for d in datasets:
        if re.search(rf"\b{re.escape(d)}\b", code, re.I):
            keywords.add(d)

    # Try AST parse to get class names and function names
    class_names, func_names = [], []
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                func_names.append(node.name)
    except Exception:
        tree = None

    # simple hyperparam extraction like epochs = 10, batch_size = 32
    simple_meta = {}
    for m in re.finditer(r"(\b(?:epochs|batch_size|lr|learning_rate)\b)\s*=\s*([0-9\.]+)", code, re.I):
        key = m.group(1).lower()
        val = m.group(2)
        simple_meta[key] = val

    # gather string tokens for heuristics
    str_tokens = []
    if tree:
        str_tokens = _extract_strings_from_ast(tree)

    # model candidate heuristics
    model_candidates = []
    for token in str_tokens + class_names + func_names:
        if re.search(r"(model|net|resnet|transformer|bert|cnn|lstm|gru|mixer)", token, re.I):
            model_candidates.append(token)

    metadata = {
        "keywords": list(keywords) + model_candidates,
        "class_names": class_names,
        "func_names": func_names,
        "simple_meta": simple_meta,
        "raw_code_snippet": code[:4000]
    }
    return metadata
