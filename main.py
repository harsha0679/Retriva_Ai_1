import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.domain.models import HealthResponse
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

app = FastAPI(
    title="KnowledgeHub AI",
    description="Enterprise Knowledge Platform — FastEmbed + ChromaDB RAG Pipeline",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    """Pre-load all singletons on the main thread before accepting requests."""
    logger = logging.getLogger("startup")
    logger.info("Initializing VectorDB and FastEmbed...")
    from app.infrastructure.embeddings.factory import EmbeddingFactory
    from app.infrastructure.vector_db.factory import VectorDBFactory
    from app.infrastructure.llm.factory import LLMFactory

    EmbeddingFactory.get_provider()
    VectorDBFactory.get_provider()
    LLMFactory.get_provider()
    logger.info("All models & DBs loaded. Server ready!")

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload_router, tags=["Phase 1: Ingestion"])
app.include_router(documents_router, tags=["Phase 2.5: Document Management"])
app.include_router(chat_router, tags=["Phase 2: RAG"])

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check."""
    vector_db_status = "ok"
    llm_status = "ok"
    try:
        from app.infrastructure.vector_db.factory import VectorDBFactory
        VectorDBFactory.get_provider().list_documents()
    except Exception:
        vector_db_status = "error"
    try:
        from app.infrastructure.llm.factory import LLMFactory
        LLMFactory.get_provider()
    except Exception:
        llm_status = "error"
    return HealthResponse(api="ok", vector_db=vector_db_status, llm=llm_status)
