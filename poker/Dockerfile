FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY poker_bot/pyproject.toml .
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY poker_bot/src/poker_bot ./poker_bot

# Environment variables
ENV PHOENIX_HOST=phoenix
ENV PHOENIX_PORT=6006
ENV PHOENIX_GRPC_PORT=4317

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nc -z localhost 6006 || exit 1

ENTRYPOINT ["python", "-m", "poker_bot.main"]
