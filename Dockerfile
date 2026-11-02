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

RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p /app/logs /app/data /app/.hf \
    && chown app:app /app/logs /app/data /app/.hf
USER 10001

# Bake the embedding model in (as the app user, so there is no chown layer copying it), so the container never
# needs the internet to start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=10001:10001 ciap ./ciap
COPY --chown=10001:10001 connections.json ./connections.json

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "ciap.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
