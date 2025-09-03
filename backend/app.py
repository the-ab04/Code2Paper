from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import ALLOWED_ORIGINS
from routes.paper_routes import router as paper_router

app = FastAPI(title="Code2Paper Backend", version="1.0.0")

# CORS for React dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "code2paper"}

# API routes
app.include_router(paper_router)#, prefix="/api/paper", tags=["paper"])
