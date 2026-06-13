# ============================================================
# Mundial 2026 Predictor — Dockerfile
# Base: Python 3.13-slim | Server: Gunicorn + UvicornWorker
# ============================================================

FROM python:3.13-slim

# Evitar prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias del sistema necesarias para catboost / lightgbm / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copiar el codigo fuente
COPY backend/   ./backend/
COPY frontend/  ./frontend/

# Copiar modelos entrenados y datos (se pueden sobreescribir con volumes)
COPY models/    ./models/
COPY data/      ./data/

# Copiar scripts de la raiz que predictor.py importa en runtime
COPY predict_mundial.py       .
COPY real_time_features.py    .
COPY feature_engineering.py   .

# Copiar variables de entorno de ejemplo (el .env real llega por volume/env)
COPY .env.example .

# Puerto que expone la API
EXPOSE 8000

# Health check: llama al endpoint /health cada 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# FastAPI es ASGI — requiere UvicornWorker, no wsgi puro
# -w 2: 2 workers (ajustar segun CPU disponible)
# --chdir /app/backend: el modulo main:app vive en backend/
CMD ["gunicorn", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--chdir", "/app/backend", \
     "main:app"]
