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
.
├── Dockerfile
├── README.md
├── logs
│   └── app.log
├── main.py
├── poetry.lock
├── pyproject.toml
├── setup.sh
├── src
│   └── nisaa
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   └── __init__.cpython-311.pyc
│       ├── api
│       │   ├── __pycache__
│       │   │   ├── chatbot_router.cpython-311.pyc
│       │   │   ├── extract_router.cpython-311.pyc
│       │   │   ├── rest_server.cpython-310.pyc
│       │   │   ├── rest_server.cpython-311.pyc
│       │   │   └── wati_router.cpython-311.pyc
│       │   ├── chatbot_router.py
│       │   ├── extract_router.py
│       │   ├── rest_server.py
│       │   └── wati_router.py
│       ├── controllers
│       │   ├── __pycache__
│       │   │   └── extract_controller.cpython-311.pyc
│       │   └── extract_controller.py
│       ├── graphs
│       │   ├── __pycache__
│       │   │   ├── graph.cpython-311.pyc
│       │   │   ├── node.cpython-311.pyc
│       │   │   └── state.cpython-311.pyc
│       │   ├── graph.py
│       │   ├── node.py
│       │   └── state.py
│       ├── helpers
│       │   ├── __pycache__
│       │   │   ├── agent_factory.cpython-311.pyc
│       │   │   ├── db.cpython-311.pyc
│       │   │   ├── logger.cpython-311.pyc
│       │   │   ├── long_term_memory.cpython-311.pyc
│       │   │   ├── short_term_memory.cpython-311.pyc
│       │   │   ├── tika_client.cpython-311.pyc
│       │   │   └── top_k_fetch.cpython-311.pyc
│       │   ├── db.py
│       │   ├── logger.py
│       │   ├── long_term_memory.py
│       │   ├── short_term_memory.py
│       │   ├── tika_client.py
│       │   └── top_k_fetch.py
│       ├── models
│       │   ├── __pycache__
│       │   │   └── extract_model.cpython-311.pyc
│       │   └── extract_model.py
│       ├── prompt
│       │   ├── __pycache__
│       │   │   └── chat_bot.cpython-311.pyc
│       │   └── chat_bot.py
│       └── services
│           ├── __pycache__
│           │   ├── pinecone_client.cpython-311.pyc
│           │   ├── top_k_fetch.cpython-311.pyc
│           │   ├── wati_api_service.cpython-311.pyc
│           │   └── wati_webhook.cpython-311.pyc
│           ├── pinecone_client.py
│           └── wati_api_service.py
└── tests
    └── __init__.py
```

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/arpitkumar20/Nissa.git
```

2. Create a virtual environment (first time only):
```bash

```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Upgrade pip:
```bash
poetry upgrade
```

5. Install dependencies:
```bash
poetry insatll
```

6. Run REST (Flask) Service:
```bash
python -m main.py
```

### NGROK

```bash
Domain : aeronautic-showier-marquitta.ngrok-free.app
ngrok http 5004 --url aeronautic-showier-marquitta.ngrok-free.app
ngrok http --url=aeronautic-showier-marquitta.ngrok-free.app 4011
```
### DB URI 
```
"postgresql://postgres:mN7pR4xT9aQ1zK6b@raising-db-instance.ci1kiooywret.us-east-1.rds.amazonaws.com:5432/postgres"
```
### TIKA SERVER

```bash
docker run -d -p 9998:9998 logicalspark/docker-tikaserver
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
