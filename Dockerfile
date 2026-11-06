# Agent Assist (intent pipeline) service: FastAPI. Build: docker build -t ciap-agent-assist .
# One process. Intent matching uses the all-MiniLM-L6-v2 embedding model, which is downloaded at build time
# and baked into the image (it then runs with HF_HUB_OFFLINE=1). The LLM (Ollama) is a separate container:
# point OLLAMA_HOST at it. Settings: CIAP_API_BASE_URL, CIAP_API_KEY, AGENT_ASSIST_API_KEY, OLLAMA_HOST,
# RESPONDER_MODEL_NAME (see ciap/config.py).

# Python 3.12: requirements.txt pins numpy 1.26.4, which has no 3.13 wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# torch first, from the CPU-only index: the default wheel pulls the CUDA libraries (several GB).
COPY requirements.txt .
RUN pip install --no-cache-dir torch==$(grep -E '^torch==' requirements.txt | cut -d= -f3) \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Runs as uid 10001 by default; any uid works with group 0 (docker run --user <uid>:0), e.g. the host user
# that owns the mounted logs/ and data/ folders. What the service writes (logs/, data/, the model cache) is
# owned by app and group 0, writable by both.
RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p /app/logs /app/data \
    && chown app:0 /app/logs /app/data && chmod g=u /app/logs /app/data

# Bake the embedding model in, so the container never needs the internet to start. The ownership change is in
# the same step, so the model isn't stored twice.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R app:0 /app/.hf && chmod -R g=u /app/.hf
USER 10001

# Code stays owned by root (read-only to the service user).
COPY ciap ./ciap
COPY connections.json ./connections.json

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "ciap.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
