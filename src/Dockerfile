FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    python3-dev \
    x11-apps \
    xauth \
    xvfb \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    scrot \
    && rm -rf /var/lib/apt/lists/*

# Install Open Interpreter with OS dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir "open-interpreter[os]"

# Create non-root user
RUN useradd -m -s /bin/bash user
USER user
WORKDIR /home/user

# Set environment variables
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1