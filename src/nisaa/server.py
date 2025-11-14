import json
import logging
import os
import uvicorn
from datetime import datetime
from typing import Any, Dict

from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.nisaa.config.db_connection import get_pool
from src.nisaa.config.db import initialize_db
from src.nisaa.router.ingestion_router import router as ingestion_router
from src.nisaa.router.wati_router import router as wati_router

from src.nisaa.sql_agent.agent import create_stateless_agent
from src.nisaa.sql_agent.agent_context import PostgresChatHistory
from src.nisaa.sql_agent.chat_manager import ChatManager

from src.nisaa.rag.node import get_rag_engine

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
    Lifespan context manager for startup and shutdown.

    Startup:
      - initialize ChatManager (agent + history)
      - initialize DB pool
      - initialize RAG engine (best-effort)

    Shutdown:
      - close DB pool connections if present
    """
    # Initialize ChatManager
    try:
        history_db = PostgresChatHistory()
        stateless_agent = create_stateless_agent()
        bot = ChatManager(agent=stateless_agent, history_manager=history_db)
        app.state.bot = bot
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
    # Initialize Database
    try:
        initialize_db()
    except Exception as e:
        logger.error("Cannot start server without database. Exiting...", exc_info=True)
        # Fail fast: raise to stop startup
        raise RuntimeError(f"Database initialization failed: {e}")
    # Initialize RAG Engine
    try:
        namespace = load_namespace()
        get_rag_engine(namespace)

    except Exception as e:
        logger.error(f"RAG Engine initialization failed: {e}", exc_info=True)

    # Hand control back to FastAPI (app starts serving)
    try:
        yield
    finally:
        logger.info("Shutting down Nisaa API Server...")
        # Graceful DB pool close if available
        try:
            pool = get_pool()
            if pool:
                pool.closeall()
                logger.info("✓ Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}", exc_info=True)


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="Nisaa API",
        description="REST API for Nisaa application with Agentic RAG",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Health Check Router
    health_router = APIRouter(prefix="/health", tags=["health"])

    @health_router.get("/")
    async def health_check() -> Dict[str, Any]:
        """Basic health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "nisaa-api",
            "version": "1.0.0",
        }

    # Include Routers
    app.include_router(health_router)
    app.include_router(wati_router, tags=["WhatsApp RAG"])
    app.include_router(ingestion_router, tags=["Ingestion"])

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    return app

# Create the FastAPI app instance
app = create_app()

# Function to run the server
def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """
    Run the FastAPI server using Uvicorn.
    """
    logger.info(f"✓ Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    # Run Uvicorn server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        access_log=True,
    )
