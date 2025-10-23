FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 4011

CMD ["python", "main.py"]
