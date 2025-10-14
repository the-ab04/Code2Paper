# backend/db/init_db.py

from .base import Base, engine
from . import models  # ensures all models are imported and registered


def init_db():
    """
    Initialize the PostgreSQL database.
    - Creates all tables if they don't exist.
    - Should be called on FastAPI startup.
    """
    print("🔧 Initializing database...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("✅ Database tables are ready.")
