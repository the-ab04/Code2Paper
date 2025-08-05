# 🧠 Code2Paper

> Automatically generate research papers from Machine Learning code using Large Language Models (LLMs)

---

## 📌 Overview

**Code2Paper** is an intelligent tool that takes your machine learning code (e.g., Python scripts or Jupyter Notebooks) and automatically generates structured research paper content — including sections like the abstract, methodology, results, and conclusion.

It leverages pre-trained LLMs such as GPT-2, T5, and StarCoder to understand ML workflows and produce coherent, academic-style text.

---

## 🚀 Features

- 🧾 Convert ML code into full research papers
- 🧠 Powered by open-source LLMs (via Hugging Face)
- 🧰 Supports both `.py` and `.ipynb` files
- 🧩 Modular output (choose abstract, method, etc.)
- 🧷 Automatic citation management (Zotero + CrossRef)
- 📄 PDF generation with proper formatting
- 💻 Web-based interface (React + Flask/FastAPI)

---

## 🛠 Tech Stack

| Component      | Technology        |
|----------------|-------------------|
| Frontend       | React.js / HTML-CSS-JS |
| Backend        | Flask or FastAPI (Python) |
| NLP Models     | Hugging Face Transformers (GPT-2, T5, StarCoder) |
| Citation Tools | Zotero + CrossRef API |
| PDF Export     | WeasyPrint / ReportLab |
| Hosting        | GitHub Pages / Google Colab / AWS (optional) |

---

## 📂 Folder Structure (Planned)
```
Code2Paper/
├── backend/                    # Python backend (Flask or FastAPI)
│   ├── app.py                  # Entry point for the backend
│   ├── routes/                 # API endpoints (e.g., /generate, /upload)
│   │   └── paper_routes.py
│   ├── services/               # Core logic for parsing code, calling LLMs, etc.
│   │   ├── code_parser.py
│   │   ├── llm_generator.py
│   │   ├── citation_manager.py
│   │   └── pdf_generator.py
│   ├── utils/                  # Utility scripts (formatting, error handling)
│   ├── templates/              # Optional: HTML templates for rendering results
│   ├── static/                 # Optional: static files if using HTML templates
│   ├── models/                 # LLM config, tokenizer, model wrapper classes
│   ├── requirements.txt        # Python dependencies
│   └── README.md

├── frontend/                   # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── pages/              # Pages like Home, Upload, Result
│   │   ├── api/                # Axios or fetch wrappers to call backend
│   │   └── App.js              # Main React app
│   ├── package.json            # Frontend dependencies
│   └── README.md

├── models/                     # Fine-tuned LLMs (optional) or config files
│   └── starcoder_config.json

├── sample_inputs/              # Example ML code (.py or .ipynb)
│   ├── sample_model.py
│   └── sample_notebook.ipynb

├── outputs/                    # Generated research papers (PDFs, .docx)
│   └── example_output.pdf

├── LICENSE                     # MIT license
├── README.md                   # Main project overview
└── .gitignore                  # Git ignore config
```

---

## 🧪 Example Workflow

1. Upload ML code file (e.g., `train_model.py`)
2. Select which sections you want to generate
3. Model extracts key components:
   - Dataset, model, training, evaluation
4. LLM generates structured academic text
5. Output PDF with formatted references is created

---

## ⚙️ Installation (Local Dev)

```bash
git clone https://github.com/the-ab04/Code2Paper.git
cd Code2Paper/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

## 📌 Project Status
🚧 In Progress


## 🙌 Contributors
Endla Akhil Balaji (@the-ab04)
Mandala Sriman Narayana
Adi Sai Kiran
Mohammad Sohel
