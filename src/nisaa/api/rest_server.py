import uvicorn
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.nisaa.api.chatbot_router import router as chatbot_router
from src.nisaa.api.extract_router import router as extract_router
from src.nisaa.api.wati_router import router as wati_router


app = FastAPI(
    title="Nisaa API",
    description="REST API for Nisaa application",
    version="0.1.0"
)

health_router = APIRouter(
    prefix="/health",
    tags=["health"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chatbot_router)


@health_router.get("/")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nisaa-api",
        "version": "0.1.0"
    }


@health_router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
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
app.include_router(wati_router)


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "Welcome to Nisaa API",
        "health_check": "/health/",
        "detailed_health": "/health/detailed",
        "docs": "/docs"
    }


def create_app() -> FastAPI:
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    uvicorn.run(
        "nisaa.api.rest_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(debug=True)
