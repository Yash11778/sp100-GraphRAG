# HuggingFace Spaces (Docker SDK) — serves the FastAPI app on port 7860.
# Python 3.14 to match the pinned, reproducible requirements.txt.
FROM python:3.14-slim

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
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
