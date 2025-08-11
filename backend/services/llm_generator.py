# backend/services/llm_generator.py
"""
LLM generator module.
Behavior:
 - If env var USE_STUB_LLM=1 -> uses a deterministic simple template generator (fast, no downloads).
 - Otherwise attempts to load a HF text-generation pipeline (gpt2 by default).
Adjust model and device as needed.
"""
import os
import logging

USE_STUB = os.environ.get("USE_STUB_LLM", "1") == "1"  # default to stub for faster dev
MODEL_NAME = os.environ.get("LLM_MODEL", "gpt2")  # change to a larger model if desired

_text_gen = None

def init_llm():
    global _text_gen
    if USE_STUB:
        logging.info("Using STUB LLM (no model download). Set USE_STUB_LLM=0 to use real HF model.")
        _text_gen = None
        return

    # Lazy import heavy libs
    try:
        from transformers import pipeline, set_seed
    except Exception as e:
        logging.error("transformers not available: %s", e)
        raise

    if _text_gen is None:
        _text_gen = pipeline("text-generation", model=MODEL_NAME, device=-1)  # device=-1 => CPU
        set_seed(42)

def _build_prompt(section, metadata):
    if section.lower() == "abstract":
        prompt = (
            "Write a concise academic abstract for a machine learning project. "
            f"Key details: keywords={metadata.get('keywords')}, hyperparams={metadata.get('simple_meta')}. "
            "Summarize objective, approach, and expected outcome in 2-4 sentences."
        )
    elif section.lower() in ("methodology", "methods"):
        prompt = (
            "Write the Methodology section describing the model architecture and training approach. "
            f"Metadata: classes={metadata.get('class_names')}, functions={metadata.get('func_names')}, hyper={metadata.get('simple_meta')}."
        )
    elif section.lower() == "results":
        prompt = (
            "Write a Results section describing how evaluation is performed. "
            "If no numeric results are available, include placeholder text describing recommended metrics and reporting style."
        )
    elif section.lower() == "conclusion":
        prompt = (
            "Write a short Conclusion summarizing contributions, limitations, and future work."
        )
    else:
        prompt = f"Write a short {section} section for an ML project."
    return prompt

def generate_section_text(prompt_data):
    section = prompt_data.get("section", "abstract")
    metadata = prompt_data.get("metadata", {})

    # If using stub, return a deterministic template
    if USE_STUB:
        # simple templated output, safe and instant
        km = ", ".join(metadata.get("keywords", [])[:5]) or "ML code"
        hm = ", ".join([f"{k}={v}" for k, v in metadata.get("simple_meta", {}).items()]) or "default hyperparameters"
        if section.lower() == "abstract":
            return (f"This work presents an implementation of {km}. "
                    f"The approach uses a standard model configuration with {hm}. "
                    "Experiments were run on the code provided and demonstrate the expected pipeline for training and evaluation.")
        elif section.lower() in ("methodology", "methods"):
            return (f"The code defines model components ({', '.join(metadata.get('class_names', [])[:3])}). "
                    "Training uses common optimization routines and standard data preprocessing. "
                    f"Hyperparameters: {hm}.")
        elif section.lower() == "results":
            return ("Results are reported using standard metrics such as accuracy and loss. "
                    "If no numeric output is present in the code, include measured test/validation metrics here.")
        elif section.lower() == "conclusion":
            return ("We provide a straightforward pipeline from data loading to model evaluation. "
                    "Future work includes more experiments and ablation studies.")
        else:
            return f"A short {section} section for the project."

    # Otherwise, use HF pipeline
    global _text_gen
    if _text_gen is None:
        init_llm()

    prompt = _build_prompt(section, metadata)
    # small generation for responsiveness; tune max_length for quality
    out = _text_gen(prompt, max_length=150, num_return_sequences=1)
    return out[0]["generated_text"]
