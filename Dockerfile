# Use a lightweight Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps (for ffmpeg if audio processing needs it)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

# Copy project files
COPY --chown=appuser:appuser . /home/appuser

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir uvicorn

EXPOSE 7860 8000

# Set environment variables for Docker
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Default to Gradio UI, switch to API with APP_MODE=api
CMD if [ "$APP_MODE" = "api" ]; then \
      uvicorn local_notebooklm.server:app --host 0.0.0.0 --port 8000 --reload; \
    else \
      python -m local_notebooklm.web_ui --port 7860; \
    fi