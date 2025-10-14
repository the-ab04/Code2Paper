# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from config import ALLOWED_ORIGINS
from routes.paper_routes import router as paper_router
from routes.user_routes import router as user_router
from db.init_db import init_db

# --- Initialize App ---
app = FastAPI(
    title="Code2Paper Backend",
    version="2.0.0",
    description="End-to-End pipeline: code → papers → citations → research draft",
)

# --- CORS Middleware (React frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup / Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    # Initialize PostgreSQL tables
    init_db()
    # If Qdrant is used, we’ll initialize the client here later
    print("✅ Code2Paper backend started, DB initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Code2Paper backend shutting down.")

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"status": "ok", "service": "Code2Paper", "version": "2.0.0"}

# --- Routers ---
# Routers already have prefixes defined in their own files,
# so no need to duplicate them here.
app.include_router(paper_router)
app.include_router(user_router)
