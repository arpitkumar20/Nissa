"""
REST Server for Nisaa API
Provides health check and other API endpoints
"""

import uvicorn
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.nisaa.api import chatbot_router
from src.nisaa.api.extract_router import router as extract_router



# Create the main FastAPI application
app = FastAPI(
    title="Nisaa API",
    description="REST API for Nisaa application",
    version="0.1.0"
)

# Create a router for health check endpoints
health_router = APIRouter(
    prefix="/health",
    tags=["health"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow your frontend or all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chatbot_router.router)


@health_router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the API is running
    
    Returns:
        Dict containing status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nisaa-api",
        "version": "0.1.0"
    }


@health_router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check endpoint with more information
    
    Returns:
        Dict containing detailed health information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nisaa-api",
        "version": "0.1.0",
        "uptime": "running",
        "database": "not_configured",
        "cache": "not_configured",
        "dependencies": {
            "fastapi": "available",
            "uvicorn": "available"
        }
    }


app.include_router(health_router)
app.include_router(extract_router)

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint
    
    Returns:
        Welcome message
    """
    return {
        "message": "Welcome to Nisaa API",
        "health_check": "/health/",
        "detailed_health": "/health/detailed",
        "docs": "/docs"
    }


def create_app() -> FastAPI:
    """
    Application factory function
    
    Returns:
        Configured FastAPI application
    """
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    uvicorn.run(
        "nisaa.api.rest_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(debug=True)
