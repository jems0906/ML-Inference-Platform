FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/

# Store HuggingFace models inside /app/.cache so appuser can access them at runtime
ENV HF_HOME=/app/.cache

# Pre-download ALL models at build time so the container starts fast
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english'); pipeline('text-generation', model='microsoft/DialoGPT-small'); print('All models cached')"

# Hand ownership of /app (including cached models) to appuser
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health')"

# Use shell form so $PORT is expanded at runtime (Render injects PORT automatically)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1