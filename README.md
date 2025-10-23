# Nisaa API

A REST API server built with FastAPI providing health check endpoints and other services.

## Features

- Health check API endpoints
- FastAPI-based REST server
- Automatic API documentation
- Modular router architecture

## Quick Start

### 1. Install Dependencies

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

Or manually:
```bash
pip install -e .
```

### 2. Run the Server

```bash
# Run using main.py
python main.py

# Or run the server module directly
python -m nisaa.api.rest_server
```

### 3. Test the API

Once the server is running, you can test the health check endpoints:

```bash
# Basic health check
curl http://localhost:4011/health/

# Detailed health check
curl http://localhost:4011/health/detailed

# Root endpoint
curl http://localhost:4011/
```

## API Endpoints

### Health Check Endpoints

- `GET /health/` - Basic health check
- `GET /health/detailed` - Detailed health check with more information

### Other Endpoints

- `GET /` - Root endpoint with API information
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Development

### Project Structure

```
Nisaa/
├── main.py                    # Main entry point
├── pyproject.toml            # Project configuration
├── README.md                 # This file
├── setup.sh                  # Setup script
├── src/
│   └── nisaa/
│       ├── __init__.py
│       └── api/
│           └── rest_server.py # FastAPI server with routers
└── tests/
    └── __init__.py
```

### Configuration

The server can be configured using environment variables:

- `HOST` - Host to bind to (default: 0.0.0.0)
- `PORT` - Port to bind to (default: 4011)
- `DEBUG` - Enable debug mode (default: false)

Create a `.env` file in the project root to set these variables:

```env
HOST=127.0.0.1
PORT=8000
DEBUG=true
```

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running the FastAPI application
- **python-dotenv**: For loading environment variables from .env file

## Health Check Response Format

### Basic Health Check (`/health/`)

```json
{
  "status": "healthy",
  "timestamp": "2025-10-23T12:00:00.000000",
  "service": "nisaa-api",
  "version": "0.1.0"
}
```

### Detailed Health Check (`/health/detailed`)

```json
{
  "status": "healthy",
  "timestamp": "2025-10-23T12:00:00.000000",
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
```
