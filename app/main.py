"""
ConvI — FastAPI Entry Point

Multimodal Conversation Intelligence API
Domain: Financial Banking
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.config import get_settings
from app.routers import conversation
from app.routers import chat
from app.storage import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"📌 Default domain: {settings.default_domain}")

    # Initialize database tables (drop + recreate, non-fatal)
    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")

    # Pre-load RAG retriever (FAISS index + bge-m3 model)
    try:
        from app.rag_engine import retriever as rag_retriever
        rag_retriever.load()
    except Exception as e:
        logger.warning(f"RAG preload skipped: {e}")

    yield
    logger.info("🛑 Shutting down ConvI API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Local-first, enterprise-grade multimodal conversation intelligence "
        "system for the financial banking domain. Accepts audio recordings or "
        "text transcripts and returns structured analytics JSON."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS (open for local dev — restrict in production) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(conversation.router)
app.include_router(chat.router)


# ── Root health-check ─────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "domain": settings.default_domain,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
