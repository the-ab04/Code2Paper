# 🧠 Code2Paper

> Automatically generate research papers from Machine Learning code using Large Language Models (LLMs)

---

## 📌 Overview

**Code2Paper** is an intelligent assistant that transforms your machine learning code (Python scripts or Jupyter Notebooks) into structured **research-style documentation**.  

It uses pre-trained **Large Language Models (LLMs)** such as GPT-2, T5, and StarCoder to analyze ML workflows and generate coherent, academic-style text, complete with citations and exportable formats.

---

## 🚀 Features

- 🧾 **Code → Research Paper**: Convert ML code into full research papers  
- 🧠 **LLM-Powered**: Uses Hugging Face models (GPT-2, T5, StarCoder)  
- 📂 **File Support**: Works with `.py` and `.ipynb` files  
- 🧩 **Modular Output**: Generate specific sections (abstract, methods, results, etc.)  
- 🔗 **Citation Management**: Zotero + CrossRef API integration  
- 📄 **Formatted PDF Export**: Auto-generate publish-ready PDFs  
- 🌐 **Web Interface**: Built with React + FastAPI  

---

## 🛠 Tech Stack

| Component      | Technology |
|----------------|------------|
| **Frontend**   | React.js, Tailwind CSS |
| **Backend**    | FastAPI (Python) |
| **NLP Models** | Hugging Face Transformers (GPT-2, T5, StarCoder) |
| **Citations**  | Zotero + CrossRef API |
| **PDF Export** | ReportLab / WeasyPrint |
| **Hosting**    | GitHub Pages, Google Colab, AWS (optional) |

---

## 📂 Project Structure

```
Code2Paper/
├── backend/ # 🔧 Backend (FastAPI service)
│ ├── app.py # 🚀 FastAPI entrypoint (run with uvicorn app:app --reload --port 8001)
│ ├── routes/ # 🌐 API route handlers (e.g., /generate, /upload)
│ │ └── paper_routes.py
│ ├── services/ # 🧠 Core logic
│ │ ├── code_parser.py
│ │ ├── llm_generator.py
│ │ ├── citation_manager.py
│ │ └── pdf_generator.py
│ ├── utils/ # 🛠 Helper utilities
│ ├── templates/ # 📄 Optional Jinja2/HTML templates
│ ├── static/ # 🎨 Optional static files
│ ├── models/ # 📦 Model configs / wrappers
│ ├── requirements.txt # 📌 Python dependencies
│ └── README.md
│
├── frontend/ # 🎨 Frontend (React.js)
│ ├── public/
│ ├── src/
│ │ ├── components/
│ │ ├── pages/
│ │ ├── api/
│ │ └── App.js
│ ├── package.json
│ └── README.md
│
├── models/ # 🧠 Pretrained/fine-tuned models or configs
│ └── starcoder_config.json
│
├── sample_inputs/ # 📝 Example ML scripts & notebooks
│ ├── sample_model.py
│ └── sample_notebook.ipynb
│
├── outputs/ # 📤 Generated research papers
│ └── example_output.pdf
│
├── LICENSE
├── README.md
└── .gitignore
```

---

## 🧪 Example Workflow

1. **Upload** your ML code (`train_model.py` or `notebook.ipynb`)  
2. **Select** sections to generate (abstract, methodology, results, conclusion)  
3. **Parse**: Extract datasets, models, training, evaluation steps  
4. **Generate**: LLM produces structured academic content  
5. **Export**: Download a formatted PDF with citations  

---

## ⚙️ Installation (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/the-ab04/Code2Paper.git

#Backend
cd Code2Paper/backend
python -m venv venv
venv\Scripts\activate   # On Windows
# or source venv/bin/activate on Linux/Mac

pip install -r requirements.txt
uvicorn app:app --reload --port 8001
# 👉 Runs on http://127.0.0.1:8001

cd ../frontend
npm install
npm start
# 👉 Runs on http://localhost:3000

```

## 📌 Project Status
🚧 In Progress


## 🙌 Contributors
- Endla Akhil Balaji (@the-ab04)
- Adi Sai Kiran (@adhi8724r)
- Mandala Sriman Narayana (@Sriman117)
- Mohammad Sohel (@sohellucky)
