# One image: React UI (built) + FastAPI backend. Works on x86-64 and ARM (e.g. Oracle Ampere).
#   docker build -t research-agent .
#   docker run -p 8080:8080 -v ra-data:/data -e ACCESS_PASSWORD=... -e GEMINI_API_KEY=... -e DEFAULT_PROVIDER=gemini research-agent

FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache TOKENIZERS_PARALLELISM=false \
    DATA_DIR=/data UPLOAD_DIR=/data/uploads WORKSPACES_DIR=/data/workspaces \
    FAISS_INDEX_DIR=/data/faiss_index REPORTS_DIR=/data/reports LOG_DIR=/data/logs DB_PATH=/data/app.db
WORKDIR /app/backend

# CPU-only PyTorch (the default wheel bundles CUDA and is several GB larger).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY --from=web /web/dist /app/frontend/dist

# Bake the embedding model into the image: fast cold starts, no download at runtime.
RUN python -m scripts.prefetch_models

RUN useradd -m -u 1000 app && mkdir -p /data && chown -R app:app /data /opt/hf-cache
USER app
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'))"
# $PORT is provided by Cloud Run and similar platforms
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
