# Serves the FastAPI app (api/app.py) directly -- no Gradio wrapper needed.
# Originally written for HF Spaces' Docker SDK (now a paid tier there); works
# unmodified on any container platform that runs a Dockerfile, e.g. Railway,
# Cloud Run, Render. Python 3.12 matches the pinned requirements and the
# Spaces metadata in README.md.
FROM python:3.12-slim

# Runtime libs: libgomp1 is needed by faiss-cpu / torch (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as a non-root user (uid 1000). Give it a writable
# HOME so model/cache downloads (fastembed, transformers, torch) have somewhere
# to go.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    XDG_CACHE_HOME=/home/user/.cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user
EXPOSE 7860
# Shell form (not exec-form CMD ["..."]) so $PORT actually expands. Railway/
# Cloud Run/Render assign a dynamic port via $PORT; HF Spaces' Docker SDK sets
# none, so it falls back to 7860 to match the README's app_port.
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}
