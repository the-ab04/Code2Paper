# backend/routes/user_routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.base import get_db
from db import crud, schemas

router = APIRouter(prefix="/api/users", tags=["users"])


# === Register a New User ===
@router.post("/register", response_model=schemas.UserOut)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user account.
    ⚠️ In production, always hash passwords before storing!
    """
    # Check if email already exists
    existing = crud.get_user_by_email(db, email=user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    return crud.create_user(db=db, user=user)


# === Get User by ID ===
@router.get("/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Fetch a user by their ID.
    """
    user = crud.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# === List All Users ===
@router.get("/", response_model=List[schemas.UserOut])
def list_users(db: Session = Depends(get_db)):
    """
    Return all users in the database.
    """
    return db.query(crud.models.User).all()
