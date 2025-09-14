# syntax=docker/dockerfile:1.6
# CPU-only backend image (no CUDA)
FROM python:3.10.14-slim

ENV PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10

# System deps (ffmpeg is often needed by video pipelines)
# NOTE: keep this as a single line on Windows to avoid CRLF + backslash issues
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (keeps cache warm)
COPY requirements.txt /tmp/requirements.txt
# Keep on one line (or use proper line breaks with LF only)
RUN --mount=type=cache,target=/root/.cache/pip pip install -U pip && pip install -r /tmp/requirements.txt

# App code
COPY . .

EXPOSE 8000
# If you have an API entrypoint, uncomment and adjust:
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
