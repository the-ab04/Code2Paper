from dotenv import load_dotenv
import os

load_dotenv()  # loads .env if present

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Keys
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
#OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# Server
PORT: int = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS: list[str] = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
