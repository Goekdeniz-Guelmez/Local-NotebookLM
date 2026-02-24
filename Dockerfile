# Use a lightweight Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system deps (for ffmpeg if audio processing needs it)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir uvicorn

EXPOSE 7860 8000

# Default to Gradio UI, switch to API with APP_MODE=api
CMD if [ "$APP_MODE" = "api" ]; then \
      uvicorn local_notebooklm.server:app --host 0.0.0.0 --port 8000 --reload; \
    else \
      python -m local_notebooklm.web_ui --port 7860; \
    fi