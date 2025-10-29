import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware


# Existing routers
from nisaa.graphs.node import get_rag_engine
from src.nisaa.api.ingestion_router import router as ingestion_router

# New RAG router
from src.nisaa.api.wati_router import router as wati_router

# Database initialization
from src.nisaa.helpers.db import initialize_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Events
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Nisaa API Server with Agentic RAG...")

    # Initialize database tables (CRITICAL - must succeed)
    try:
        logger.info("📊 Initializing database...")
        initialize_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        logger.error("🛑 Cannot start server without database. Exiting...")
        raise RuntimeError(f"Database initialization failed: {e}")

    # Initialize RAG Engine (Optional - can continue without it)
    try:
        logger.info("🤖 Initializing RAG Engine...")
        get_rag_engine()
        logger.info("✅ RAG Engine pre-loaded successfully")
    except Exception as e:
        logger.error(f"❌ RAG Engine initialization failed: {e}")
        logger.warning("⚠️ Continuing without RAG engine - queries may fail")

    logger.info("✅ Application startup complete")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Nisaa API Server...")
    
    # Close database connections
    try:
        from src.nisaa.helpers.db import get_pool
        pool = get_pool()
        if pool:
            pool.closeall()
            logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"⚠️ Error closing database connections: {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Nisaa API",
    description="REST API for Nisaa application with Agentic RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Health check router
health_router = APIRouter(prefix="/health", tags=["health"])

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoints
# ============================================================================


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

    # Check RAG engine status
    rag_status = "not_initialized"
    try:
        engine = get_rag_engine()
        if engine:
            rag_status = "healthy"
    except Exception as e:
        rag_status = f"error: {str(e)}"

    # Check database status
    db_status = "not_configured"
    try:
        from src.nisaa.helpers.db import get_pool

        pool = get_pool()
        if pool:
            # Test connection
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


# ============================================================================
# Root Endpoint
# ============================================================================


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


# ============================================================================
# Include All Routers
# ============================================================================

# Health check
app.include_router(health_router)

# New Agentic RAG WhatsApp Router
app.include_router(wati_router, tags=["WhatsApp RAG"])

# Existing routers
# app.include_router(chatbot_router, tags=["Chatbot"])
# app.include_router(extract_router, tags=["Extraction"])
# app.include_router(zoho_router, tags=["Zoho"])
app.include_router(ingestion_router, tags=["Ingestion"])


# ============================================================================
# Global Exception Handler
# ============================================================================


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


# ============================================================================
# App Factory
# ============================================================================


def create_app() -> FastAPI:
    """Factory function to create the app"""
    return app


# ============================================================================
# Server Runner
# ============================================================================


def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode with auto-reload
    """
    logger.info(f"🚀 Starting server on {host}:{port}")
    logger.info(f"📝 Debug mode: {debug}")

    uvicorn.run(
        "nisaa.api.rest_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    run_server(debug=True)