# Optional: only needed if you deploy to Render as a Docker service rather
# than a native Python service. `render.yaml` uses the native runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Render injects PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /srv
USER appuser

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
