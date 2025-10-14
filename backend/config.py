from dotenv import load_dotenv
import os

load_dotenv()  # loads .env if present

# -------------------------------------------------------------------
# 🔹 LLM Provider
# -------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# API Keys
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# -------------------------------------------------------------------
# 🔹 Server
# -------------------------------------------------------------------
PORT: int = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS: list[str] = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",") if o.strip()
]

# -------------------------------------------------------------------
# 🔹 PostgreSQL Database
# -------------------------------------------------------------------
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "code2paper_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# -------------------------------------------------------------------
# 🔹 Qdrant Vector Database
# -------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "paper_chunks")

# -------------------------------------------------------------------
# 🔹 File Storage Paths
# -------------------------------------------------------------------
STORAGE_DIR = os.getenv("STORAGE_DIR", "backend/storage")
PAPERS_DIR = os.path.join(STORAGE_DIR, "papers")   # Downloaded PDFs
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs") # Generated DOCX/PDF
INDEX_DIR = os.path.join(STORAGE_DIR, "indexes")   # FAISS/Qdrant indexes

# Ensure storage dirs exist
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
