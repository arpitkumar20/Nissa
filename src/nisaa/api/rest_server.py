import json
import os
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from ..models.agent import create_stateless_agent
from ..models.agent_context import PostgresChatHistory
from ..models.chat_manager import ChatManager

from src.nisaa.graphs.node import get_rag_engine
from src.nisaa.api.ingestion_router import router as ingestion_router

from src.nisaa.api.wati_router import router as wati_router

from src.nisaa.helpers.db import initialize_db

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_namespace() -> str:
    """Load company namespace from web_info/web_info.json"""
    folder_path = "web_info"
    filename = "web_info.json"
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                namespace = data.get("namespace") or data.get("company_namespace")
                if namespace:
                    return namespace
            except Exception as e:
                logger.error(f"Error loading namespace from file: {e}")
    logger.error("Namespace not found, defaulting to 'default'")
    return "default"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events
    """
    try:
        history_db = PostgresChatHistory()

        stateless_agent = create_stateless_agent()

        bot = ChatManager(agent=stateless_agent, history_manager=history_db)

        app.state.bot = bot

    except Exception as e:
       logger.error(f"Bot initialization failed: {e}")
    try:

        initialize_db()

    except Exception as e:
        logger.error("Cannot start server without database. Exiting...")
        raise RuntimeError(f"Database initialization failed: {e}")

    try:
        namespace = load_namespace()

        get_rag_engine(namespace)

    except Exception as e:
        logger.error(f"RAG Engine initialization failed: {e}")
        logger.warning("Continuing without RAG engine - queries may fail")

    yield

    logger.info("Shutting down Nisaa API Server...")
    
    try:
        from src.nisaa.helpers.db import get_pool
        pool = get_pool()
        if pool:
            pool.closeall()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

app = FastAPI(
    title="Nisaa API",
    description="REST API for Nisaa application with Agentic RAG",
    version="1.0.0",
    lifespan=lifespan,
)

health_router = APIRouter(prefix="/health", tags=["health"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@health_router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nisaa-api",
        "version": "1.0.0",
    }


@health_router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status"""

    rag_status = "not_initialized"
    try:
        namespace = load_namespace()
        engine = get_rag_engine(namespace)
        if engine:
            rag_status = "healthy"
    except Exception as e:
        rag_status = f"error: {str(e)}"

    db_status = "not_configured"
    try:
        from src.nisaa.helpers.db import get_pool

        pool = get_pool()
        if pool:
            conn = pool.getconn()
            pool.putconn(conn)
            db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nisaa-api",
        "version": "1.0.0",
        "uptime": "running",
        "components": {
            "database": db_status,
            "rag_engine": rag_status,
            "fastapi": "available",
            "uvicorn": "available",
        },
        "features": {
            "agentic_rag": "enabled",
            "whatsapp_integration": "enabled",
            "zoho_integration": "enabled",
            "document_extraction": "enabled",
        },
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Nisaa API with Agentic RAG",
        "version": "1.0.0",
        "health_check": "/health/",
        "detailed_health": "/health/detailed",
        "docs": "/docs",
        "endpoints": {
            "whatsapp": "/wati_webhook",
            "chatbot": "/chatbot/",
            "zoho": "/zoho/",
            "extraction": "/extract/",
            "ingestion": "/ingest/",
        },
    }

app.include_router(health_router)
app.include_router(wati_router, tags=["WhatsApp RAG"])
app.include_router(ingestion_router, tags=["Ingestion"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": str(request.url),
    }

def create_app() -> FastAPI:
    """Factory function to create the app"""
    return app

def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode with auto-reload
    """
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")

    uvicorn.run(
        "nisaa.api.rest_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        access_log=True,
    )
