from sqlalchemy.orm import Session
from . import models, schemas

# === User CRUD ===
def create_user(db: Session, user: schemas.UserCreate):
    """Register a new user (⚠️ hash password in production)."""
    db_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=user.password,  # TODO: replace with hashed password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


# === Run CRUD ===
def create_run(db: Session, input_file: str, status: str = "pending", user_id: int | None = None):
    """Create a new run when a notebook/code file is uploaded."""
    db_run = models.Run(
        input_file=input_file,
        status=status,
        user_id=user_id,
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run


def update_run_status(db: Session, run_id: int, status: str, output_file: str | None = None):
    """Update run status and optionally attach output file path."""
    run = db.query(models.Run).filter(models.Run.id == run_id).first()
    if not run:
        return None
    run.status = status
    if output_file:
        run.output_file = output_file
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: int):
    return db.query(models.Run).filter(models.Run.id == run_id).first()


# === Paper CRUD ===
def add_paper(db: Session, run_id: int, paper: schemas.PaperBase):
    """Add a discovered paper (CrossRef/arXiv/etc.) linked to a run.
       Skip insertion if DOI already exists."""
    if paper.doi:
        existing = db.query(models.Paper).filter(models.Paper.doi == paper.doi).first()
        if existing:
            return existing

    db_paper = models.Paper(
        run_id=run_id,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        venue=paper.venue,
        doi=paper.doi,
        url=paper.url,
        pdf_path=paper.pdf_path,
    )
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper


def get_papers_by_run(db: Session, run_id: int):
    return db.query(models.Paper).filter(models.Paper.run_id == run_id).all()


# === Citation CRUD ===
def add_citation(db: Session, run_id: int, paper_id: int, citation: schemas.CitationBase):
    """Insert a citation (context snippet + index) linked to paper + run.
       Skip duplicates."""
    existing = db.query(models.Citation).filter(
        models.Citation.run_id == run_id,
        models.Citation.paper_id == paper_id,
        models.Citation.index == citation.index
    ).first()
    if existing:
        return existing

    db_cite = models.Citation(
        run_id=run_id,
        paper_id=paper_id,
        context=citation.context,
        index=citation.index,
    )
    db.add(db_cite)
    db.commit()
    db.refresh(db_cite)
    return db_cite


def get_citations_by_paper(db: Session, paper_id: int):
    return db.query(models.Citation).filter(models.Citation.paper_id == paper_id).all()


def get_citations_by_run(db: Session, run_id: int):
    return db.query(models.Citation).filter(models.Citation.run_id == run_id).all()


# === Chunk CRUD ===
def add_chunk(db: Session, paper_id: int, chunk_id: int, text: str):
    """Store a text chunk linked to a paper (raw text mirrors what’s in Qdrant).
       Skip duplicates."""
    existing = db.query(models.Chunk).filter(
        models.Chunk.paper_id == paper_id,
        models.Chunk.chunk_id == chunk_id
    ).first()
    if existing:
        return existing

    db_chunk = models.Chunk(
        paper_id=paper_id,
        chunk_id=chunk_id,
        text=text,
    )
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk


def get_chunks_by_paper(db: Session, paper_id: int):
    return db.query(models.Chunk).filter(models.Chunk.paper_id == paper_id).all()
