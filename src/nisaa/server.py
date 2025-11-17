import asyncio
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

shutdown_in_progress = False


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
    Lifespan context manager for startup and shutdown with GRACEFUL handling
    """
    try:
        history_db = PostgresChatHistory()
        stateless_agent = create_stateless_agent()
        bot = ChatManager(agent=stateless_agent, history_manager=history_db)
        app.state.bot = bot
        logger.info("✓ ChatManager initialized")
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
    
    try:
        initialize_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error("Cannot start server without database. Exiting...", exc_info=True)
        raise RuntimeError(f"Database initialization failed: {e}")
    
    try:
        namespace = load_namespace()
        print("✓ Loaded namespace:", namespace)
        get_rag_engine(namespace)
        logger.info("✓ RAG engine initialized")
    except Exception as e:
        logger.error(f"RAG Engine initialization failed: {e}", exc_info=True)

    try:
        yield
    finally:

        global shutdown_in_progress
        shutdown_in_progress = True
        
        logger.info(">> Shutting down Nisaa API Server...")
        
        try:
            from main import background_tasks_running
            if background_tasks_running:
                logger.info(f">> Signaling {len(background_tasks_running)} background task(s) to stop...")
                
                for task in background_tasks_running:
                    if not task.done():
                        task.cancel()
                
                logger.info(">> Waiting up to 15 seconds for graceful shutdown...")
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*background_tasks_running, return_exceptions=True),
                        timeout=15.0
                    )
                    logger.info("✓ All background tasks stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(" Some tasks didn't finish within 15 seconds")
                    logger.warning(" Checkpoints were saved - you can resume by restarting ingestion")
                    
                    still_running = [t for t in background_tasks_running if not t.done()]
                    if still_running:
                        logger.warning(f" {len(still_running)} task(s) still running")
                    
        except ImportError:
            logger.warning("Could not import background task tracking")
        except Exception as e:
            logger.error(f"Error during task cancellation: {e}")
        
        try:
            pool = get_pool()
            if pool and not pool.closed:
                await asyncio.sleep(1)
                pool.closeall()
                logger.info("✓ Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")
        
        logger.info(" Server shutdown sequence complete")
        logger.info(" Interrupted jobs have saved checkpoints")
        logger.info(" Resume by calling the ingestion endpoint again with same company name")

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

    app.include_router(health_router)
    app.include_router(wati_router, tags=["WhatsApp RAG"])
    app.include_router(ingestion_router, tags=["Ingestion"])

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

app = create_app()

def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """
    Run the FastAPI server using Uvicorn.
    """
    logger.info(f"✓ Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        access_log=True,
    )