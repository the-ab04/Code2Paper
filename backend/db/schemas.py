from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime


# === User ===
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str  # plain password (⚠️ hash before saving)


class UserOut(UserBase):
    id: int

    class Config:
        orm_mode = True


# === Citation ===
class CitationBase(BaseModel):
    context: Optional[str] = None
    index: int


class CitationOut(CitationBase):
    id: int
    run_id: int
    paper_id: int

    class Config:
        orm_mode = True


# === Chunk ===
class ChunkBase(BaseModel):
    chunk_id: int
    text: str


class ChunkOut(ChunkBase):
    id: int
    paper_id: int

    class Config:
        orm_mode = True


# === Paper ===
class PaperBase(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_path: Optional[str] = None  # local PDF storage path


class PaperOut(PaperBase):
    id: int
    run_id: int
    citations: List[CitationOut] = []
    chunks: List[ChunkOut] = []

    class Config:
        orm_mode = True


# === Run ===
class RunBase(BaseModel):
    status: str = "pending"
    input_file: Optional[str] = None
    output_file: Optional[str] = None


class RunOut(RunBase):
    id: int
    created_at: datetime
    user_id: Optional[int] = None
    papers: List[PaperOut] = []
    citations: List[CitationOut] = []  # ✅ direct citations linked to this run

    class Config:
        orm_mode = True
