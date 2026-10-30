# Build context is the project root (see docker-compose.yml) so scripts/ and data/samples are available.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARA_ROOT=/srv/cara

WORKDIR /srv/cara

# CPU-only PyTorch keeps the image small; use a CUDA base image + default torch wheel for GPU.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r backend/requirements.txt

COPY backend backend
COPY scripts scripts
COPY data/samples data/samples

WORKDIR /srv/cara/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
