# backend/db/base.py

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# ✅ Load .env variables
load_dotenv()

# ✅ Build Database URL (PostgreSQL with psycopg2 driver)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'qwerty')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', 5432)}/"
    f"{os.getenv('POSTGRES_DB', 'code2paper_db')}"
)

# ✅ Create Engine
# echo=True → logs all SQL queries (good for debugging, disable in production)
engine = create_engine(DATABASE_URL, echo=True, future=True)

# ✅ Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

# ✅ Base class for ORM models
Base = declarative_base()

# ✅ Dependency for FastAPI routes
def get_db():
    """
    FastAPI dependency that provides a SQLAlchemy session.
    Ensures session is closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
