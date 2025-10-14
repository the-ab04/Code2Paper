# backend/services/code_parser.py

import nbformat
from nbformat.reader import NotJSONError
import re
from typing import Dict, List, Optional, Tuple


def _detect_methods_and_frameworks(text: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Heuristic extraction of method families, frameworks, and a short task label from text.
    Returns (methods, frameworks, task).
    """
    methods = set()
    frameworks = set()
    task = None

    text_l = text.lower()

    # --- Frameworks ---
    if re.search(r"\btorch\b|\btorchvision\b|\bpytorch\b", text_l):
        frameworks.add("PyTorch")
    if re.search(r"\btensorflow\b|\bkeras\b", text_l):
        frameworks.add("TensorFlow/Keras")
    if re.search(r"\bsklearn\b|\bscikit-learn\b", text_l):
        frameworks.add("scikit-learn")
    if re.search(r"\bmxnet\b", text_l):
        frameworks.add("MXNet")
    if re.search(r"\bflax\b|\bjax\b", text_l):
        frameworks.add("JAX/Flax")

    # --- Model families / methods ---
    if re.search(r"\bconv2d\b|\bconv\b|\bcnn\b|\bresnet\b|\bvgg\b|\bmobilenet\b|\binception\b", text_l):
        methods.add("Convolutional Neural Networks (CNNs)")

    if re.search(r"\btransformer\b|\battention\b|\bbert\b|\bgpt\b|\bself-attention\b", text_l):
        methods.add("Transformer / Attention-based models")

    if re.search(r"\brnn\b|\blstm\b|\bgru\b", text_l):
        methods.add("Recurrent Neural Networks (LSTM/GRU)")

    if re.search(r"\brandom forest\b|\brf\b|\bgradient boosting\b|\bxgboost\b|\bsvc\b|\bsupport vector\b", text_l):
        if re.search(r"\brandom forest\b|\brf\b", text_l):
            methods.add("Random Forest")
        if re.search(r"\bxgboost\b|\bgradient boosting\b", text_l):
            methods.add("Gradient Boosting / XGBoost")
        if re.search(r"\bsvc\b|\bsupport vector\b", text_l):
            methods.add("Support Vector Machines (SVM)")

    if re.search(r"\btransfer learning\b|\bfine-?tun", text_l):
        methods.add("Transfer learning / Fine-tuning")

    if re.search(r"\bgan\b|\bgenerative adversarial\b", text_l):
        methods.add("Generative Adversarial Networks (GANs)")

    if re.search(r"\bgnn\b|\bgraph neural\b|\bgraphconv\b|\bmessage passing\b", text_l):
        methods.add("Graph Neural Networks (GNNs)")

    if re.search(r"\bbayesian\b|\bprobabilistic\b|\bmcmc\b", text_l):
        methods.add("Bayesian / Probabilistic methods")

    if re.search(r"\bstochastic gradient\b|\bsgd\b|\badam\b", text_l):
        methods.add("Stochastic optimization (SGD/Adam)")

    # --- Task detection (simple heuristics) ---
    if re.search(r"\bimage classification\b|\bclassification\b|\bclassify\b", text_l):
        task = "image classification" if "image" in text_l else "classification"
    elif re.search(r"\bobject detection\b|\bbox\b|\bbounding box\b|\byolo\b|\brcnn\b", text_l):
        task = "object detection"
    elif re.search(r"\bsemantic segmentation\b|\bsegmentation\b|\bmask r-cnn\b", text_l):
        task = "semantic segmentation"
    elif re.search(r"\btime[- ]?series\b|\bforecast\b|\bforecasting\b", text_l):
        task = "time-series forecasting"
    elif re.search(r"\bnlp\b|\bnatural language\b|\btext classification\b|\bnamed entity\b|\btranslation\b", text_l):
        task = "natural language processing"
    elif re.search(r"\bregression\b|\bmean squared error\b|\bmae\b|\brmse\b", text_l):
        task = "regression"
    elif re.search(r"\brecommendation\b|\bcollaborative filtering\b|\bmatrix factorization\b", text_l):
        task = "recommendation"

    return list(methods), list(frameworks), task


def parse_notebook(nb_path: str) -> Dict[str, List[str]]:
    """
    Parse a Jupyter notebook (.ipynb) and extract useful facts.

    Returns a dictionary with keys:
      - "markdown": List[str] Markdown cells
      - "code": List[str] Code cells
      - "logs": List[str] Detected metric/log lines (accuracy/loss/... )
      - "datasets": List[str] Detected dataset mentions (e.g., CIFAR, MNIST)
      - "metrics": List[str] Detected metric keywords (accuracy, loss, etc.)
      - "methods": List[str] Detected method families (CNNs, Transformer, etc.)
      - "frameworks": List[str] Detected frameworks (PyTorch, TensorFlow, scikit-learn)
      - "task": Optional[str] Short task label (classification, regression, etc.)
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
        source = cell.get("source", "") or ""
        source = source.strip()

        if not source:
            continue

        if cell_type == "markdown":
            md_blocks.append(source)

        elif cell_type == "code":
            code_blocks.append(source)

            # detect metric-like output lines inside code (simple heuristics)
            for line in source.splitlines():
                if re.search(r"(accuracy|acc|loss|f1|precision|recall|auc)\s*[:=]\s*\d+(\.\d+)?", line, re.I):
                    logs.append(line.strip())

    # Combined text for heuristics
    combined_text = "\n".join(md_blocks + code_blocks + logs)

    # Simple dataset detection (expandable)
    known_datasets = [
        "MNIST", "CIFAR-10", "CIFAR", "IMDB", "ImageNet", "COCO", "SQuAD", "GLUE", "LibriSpeech"
    ]
    datasets = [ds for ds in known_datasets if re.search(r"\b" + re.escape(ds) + r"\b", combined_text, re.I)]

    # Common metric keywords
    metrics = [
        m for m in ["accuracy", "precision", "recall", "f1", "auc", "loss", "mse", "rmse"]
        if re.search(r"\b" + re.escape(m) + r"\b", combined_text, re.I)
    ]

    # Methods, frameworks and task inference
    methods, frameworks, task = _detect_methods_and_frameworks(combined_text)

    return {
        "markdown": md_blocks,
        "code": code_blocks,
        "logs": logs,
        "datasets": datasets,
        "metrics": metrics,
        "methods": methods,
        "frameworks": frameworks,
        "task": task,
    }


if __name__ == "__main__":
    # simple local test helper
    test_path = "sample_inputs/sample_notebook.ipynb"
    try:
        facts = parse_notebook(test_path)
        print("Notebook facts extracted:")
        for key, value in facts.items():
            if isinstance(value, list):
                preview = value[:3]
                print(f"{key} ({len(value)}): {preview} ...")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
