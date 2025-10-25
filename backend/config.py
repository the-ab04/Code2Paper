# backend/config.py
from dotenv import load_dotenv
import os

# Load .env file into environment (no error if .env absent)
load_dotenv()

# -------------------------------------------------------------------
# 🔹 LLM Provider & Model settings
# -------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# API Keys (keep these secret and configure in environment or CI secrets)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Optional: keep for future use if you use OpenAI or other providers
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model and inference parameters
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.15"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

# -------------------------------------------------------------------
# 🔹 Embedding & Retriever configuration
# -------------------------------------------------------------------
# SentenceTransformers embedding model (used in rag_retriever)
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL",
    "krlvi/sentence-t5-base-nlpl-code_search_net"
)

# Qdrant connection can be defined via URL/API key or host/port
QDRANT_URL: str = os.getenv("QDRANT_URL", os.getenv("QDRANT_HOST", "http://localhost:6333"))
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "code2paper_chunks")

# Upsert batch size for embeddings (embedding / upsert batching)
BATCH_UPSERT_SIZE: int = int(os.getenv("QDRANT_UPSERT_BATCH", os.getenv("BATCH_UPSERT_SIZE", "128")))

# Score threshold for retrieval filtering (tuneable)
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.20"))

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
# 🔹 Unpaywall (Open Access lookup)
# -------------------------------------------------------------------
UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", "")  # required by Unpaywall API to be polite

# -------------------------------------------------------------------
# 🔹 File Storage Paths (used by various modules)
# -------------------------------------------------------------------
# You might prefer 'storage' at root (matches many service files using 'storage/...')
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
PAPERS_DIR = os.path.join(STORAGE_DIR, "papers")   # Downloaded PDFs
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs") # Generated DOCX/PDF
INDEX_DIR = os.path.join(STORAGE_DIR, "indexes")   # FAISS/Qdrant index files (optional)

# Ensure storage dirs exist at runtime
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 🔹 Misc / Safety
# -------------------------------------------------------------------
# Use lower-level debug toggles as needed
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
