from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    func,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    """User accounts (optional, only if you enable authentication)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)  # ⚠️ store hashed password

    # Relationships
    runs = relationship("Run", back_populates="user", cascade="all, delete-orphan")


class Run(Base):
    """Each generation attempt (upload → output paper)."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="pending")

    input_file = Column(String(500))   # path to uploaded notebook/code
    output_file = Column(String(500))  # path to generated DOCX/PDF

    # Relationships
    user = relationship("User", back_populates="runs")
    papers = relationship("Paper", back_populates="run", cascade="all, delete-orphan")
    citations = relationship("Citation", back_populates="run", cascade="all, delete-orphan")


class Paper(Base):
    """Metadata of external papers discovered and linked to a run."""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    title = Column(Text, nullable=False)
    authors = Column(Text, nullable=True)
    year = Column(Integer, nullable=True)
    venue = Column(String(255), nullable=True)
    doi = Column(String(255), nullable=True)
    url = Column(String(500), nullable=True)
    pdf_path = Column(String(500), nullable=True)  # stored OA PDF path

    # Ensure uniqueness of DOI across papers
    __table_args__ = (UniqueConstraint("doi", name="uq_papers_doi"),)

    # Relationships
    run = relationship("Run", back_populates="papers")
    citations = relationship("Citation", back_populates="paper", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")


class Citation(Base):
    """Inline citation resolved during generation (maps snippet → paper)."""
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)

    context = Column(Text, nullable=True)  # sentence or section snippet where used
    index = Column(Integer, nullable=False)  # [1], [2], ...

    # Relationships
    run = relationship("Run", back_populates="citations")
    paper = relationship("Paper", back_populates="citations")


class Chunk(Base):
    """Optional: keep raw text of chunks aligned with Qdrant embeddings."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)

    chunk_id = Column(Integer, nullable=False)  # matches Qdrant payload "chunk_id"
    text = Column(Text, nullable=False)

    # Relationships
    paper = relationship("Paper", back_populates="chunks")
