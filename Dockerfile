# Crop Master Agronomist API + web frontend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first, so code changes don't reinstall them
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY frontend frontend
# Reference data only (crop_master.xlsx, regional_docs). Databases and orders
# are excluded by .dockerignore and live in the mounted /srv/data volume.
COPY data data

RUN useradd --create-home --uid 1000 app && chown -R app:app /srv/data
USER app

WORKDIR /srv/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# One worker: SQLite stores and in-memory state (rate limits, loaded workbook) are per process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
