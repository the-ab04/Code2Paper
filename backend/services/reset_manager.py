# backend/services/reset_manager.py

import os
import shutil
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from db import models

# Load Qdrant info from environment or config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "paper_chunks")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

STORAGE_DIR = "storage/papers"


def reset_database(db: Session):
    """Delete all runs, papers, citations, chunks."""
    db.query(models.Chunk).delete()
    db.query(models.Citation).delete()
    db.query(models.Paper).delete()
    db.query(models.Run).delete()
    db.commit()
    print("✅ Database tables cleared.")


def reset_qdrant_collection():
    """Delete existing Qdrant collection."""
    try:
        if qdrant.collection_exists(QDRANT_COLLECTION):
            qdrant.delete_collection(QDRANT_COLLECTION)
            print(f"✅ Qdrant collection '{QDRANT_COLLECTION}' deleted.")
    except Exception as e:
        print(f"[Qdrant Error] Failed to delete collection: {e}")


def reset_storage():
    """Delete all files in storage/papers."""
    try:
        if os.path.exists(STORAGE_DIR):
            shutil.rmtree(STORAGE_DIR)
            os.makedirs(STORAGE_DIR)
            print(f"✅ Storage folder '{STORAGE_DIR}' cleared.")
    except Exception as e:
        print(f"[Storage Error] Failed to clear folder: {e}")


def full_reset(db: Session):
    """Perform full reset: DB, Qdrant, storage."""
    reset_database(db)
    reset_qdrant_collection()
    reset_storage()
