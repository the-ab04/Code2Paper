# backend/routes/auth_routes.py
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from passlib.context import CryptContext

from db.base import get_db
from db import crud, schemas  # crud.create_user, crud.get_user_by_email, schemas.UserCreate, schemas.UserOut maybe existing

# Config (pull from env or defaults)
SECRET_KEY = os.getenv("JWT_SECRET", "supersecret-dev-key")  # change in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["auth"])


# Pydantic models (fallback if you don't have matching ones in db/schemas)
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SignUpIn(BaseModel):
    name: Optional[str] = None
    email: EmailStr
    password: str


class UserInfo(BaseModel):
    id: int
    username: Optional[str] = None
    email: EmailStr


# Helper utilities
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token


# Signup endpoint
@router.post("/signup", response_model=UserInfo)
def signup(payload: SignUpIn, db: Session = Depends(get_db)):
    # if your schemas.UserCreate expects different fields, adapt here
    existing = crud.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # create user (crud.create_user may be implemented; it currently stores password as password_hash field)
    user_create_schema = schemas.UserCreate(username=payload.name or payload.email.split("@")[0],
                                            email=payload.email,
                                            password=hash_password(payload.password))
    db_user = crud.create_user(db, user_create_schema)

    return UserInfo(id=db_user.id, username=db_user.username, email=db_user.email)


# Login endpoint (OAuth2PasswordRequestForm expects username/password keys)
@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm provides .username and .password
    user = crud.get_user_by_email(db, form_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # crud.create_user stored password_hash in `password_hash` column; adjust naming if different
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    return Token(access_token=access_token, token_type="bearer")


# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = crud.get_user(db, int(user_id))
    if user is None:
        raise credentials_exception
    return user


@router.get("/me", response_model=UserInfo)
def read_me(current_user=Depends(get_current_user)):
    return UserInfo(id=current_user.id, username=current_user.username, email=current_user.email)
